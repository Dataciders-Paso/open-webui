import asyncio
import base64
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from open_webui.config import CACHE_DIR
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, SRC_LOG_LEVELS
from open_webui.routers.files import upload_file
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.videos.comfyui import (
    ComfyUIGenerateVideoForm,
    ComfyUIWorkflow,
    comfyui_generate_video,
)
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["VIDEOS"])

VIDEO = CACHE_DIR / "video" / "generations"
VIDEO.mkdir(parents=True, exist_ok=True)


router = APIRouter()


@router.get("/videos/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "enabled": request.app.state.config.ENABLE_VIDEO_GENERATION,
        "engine": request.app.state.config.VIDEO_GENERATION_ENGINE,
        "prompt_generation": request.app.state.config.ENABLE_VIDEO_PROMPT_GENERATION,
        "comfyui": {
            "COMFYUI_BASE_URL": request.app.state.config.COMFYUI_BASE_URL,
            "COMFYUI_API_KEY": request.app.state.config.COMFYUI_API_KEY,
            "COMFYUI_WORKFLOW": request.app.state.config.COMFYUI_WORKFLOW,
            "COMFYUI_WORKFLOW_NODES": request.app.state.config.COMFYUI_WORKFLOW_NODES,
        },
    }

class ComfyUIConfigForm(BaseModel):
    COMFYUI_BASE_URL: str
    COMFYUI_API_KEY: str
    COMFYUI_WORKFLOW: str
    COMFYUI_WORKFLOW_NODES: list[dict]


class ConfigForm(BaseModel):
    enabled: bool
    engine: str
    prompt_generation: bool
    comfyui: ComfyUIConfigForm


@router.post("/videos/config/update")
async def update_config(
    request: Request, form_data: ConfigForm, user=Depends(get_admin_user)
):
    request.app.state.config.VIDEO_GENERATION_ENGINE = form_data.engine
    request.app.state.config.ENABLE_VIDEO_GENERATION = form_data.enabled

    request.app.state.config.ENABLE_VIDEO_PROMPT_GENERATION = (
        form_data.prompt_generation
    )

    request.app.state.config.COMFYUI_BASE_URL = (
        form_data.comfyui.COMFYUI_BASE_URL.strip("/")
    )
    request.app.state.config.COMFYUI_API_KEY = form_data.comfyui.COMFYUI_API_KEY

    request.app.state.config.COMFYUI_WORKFLOW = form_data.comfyui.COMFYUI_WORKFLOW
    request.app.state.config.COMFYUI_WORKFLOW_NODES = (
        form_data.comfyui.COMFYUI_WORKFLOW_NODES
    )

    return {
        "enabled": request.app.state.config.ENABLE_VIDEO_GENERATION,
        "engine": request.app.state.config.VIDEO_GENERATION_ENGINE,
        "prompt_generation": request.app.state.config.ENABLE_VIDEO_PROMPT_GENERATION,
        "comfyui": {
            "COMFYUI_BASE_URL": request.app.state.config.COMFYUI_BASE_URL,
            "COMFYUI_API_KEY": request.app.state.config.COMFYUI_API_KEY,
            "COMFYUI_WORKFLOW": request.app.state.config.COMFYUI_WORKFLOW,
            "COMFYUI_WORKFLOW_NODES": request.app.state.config.COMFYUI_WORKFLOW_NODES,
        },
    }


def get_automatic1111_api_auth(request: Request):
    if request.app.state.config.AUTOMATIC1111_API_AUTH is None:
        return ""
    else:
        auth1111_byte_string = request.app.state.config.AUTOMATIC1111_API_AUTH.encode(
            "utf-8"
        )
        auth1111_base64_encoded_bytes = base64.b64encode(auth1111_byte_string)
        auth1111_base64_encoded_string = auth1111_base64_encoded_bytes.decode("utf-8")
        return f"Basic {auth1111_base64_encoded_string}"


@router.get("/videos/config/url/verify")
async def verify_url(request: Request, user=Depends(get_admin_user)):
    if request.app.state.config.VIDEO_GENERATION_ENGINE == "comfyui":

        headers = None
        if request.app.state.config.COMFYUI_API_KEY:
            headers = {
                "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
            }

        try:
            r = requests.get(
                url=f"{request.app.state.config.COMFYUI_BASE_URL}/object_info",
                headers=headers,
            )
            r.raise_for_status()
            return True
        except Exception:
            request.app.state.config.ENABLE_VIDEO_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)
    else:
        return True


def set_video_model(request: Request, model: str):
    log.info(f"Setting video model to {model}")
    request.app.state.config.VIDEO_GENERATION_MODEL = model
    if request.app.state.config.VIDEO_GENERATION_ENGINE in [""]:
        api_auth = get_automatic1111_api_auth(request)
        r = requests.get(
            url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
            headers={"authorization": api_auth},
        )
        options = r.json()
        if model != options["sd_model_checkpoint"]:
            options["sd_model_checkpoint"] = model
            r = requests.post(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                json=options,
                headers={"authorization": api_auth},
            )
    return request.app.state.config.VIDEO_GENERATION_MODEL


def get_video_model(request):
    if request.app.state.config.VIDEO_GENERATION_ENGINE == "comfyui":
        return (
            request.app.state.config.VIDEO_GENERATION_MODEL
            if request.app.state.config.VIDEO_GENERATION_MODEL
            else ""
        )
    elif (
        request.app.state.config.VIDEO_GENERATION_ENGINE == ""
    ):
        try:
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/options",
                headers={"authorization": get_automatic1111_api_auth(request)},
            )
            options = r.json()
            return options["sd_model_checkpoint"]
        except Exception as e:
            request.app.state.config.ENABLE_VIDEO_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class VideoConfigForm(BaseModel):
    MODEL: str
    VIDEO_SIZE: str
    VIDEO_STEPS: int


@router.get("/video/config")
async def get_video_config(request: Request, user=Depends(get_admin_user)):
    return {
        "MODEL": request.app.state.config.VIDEO_GENERATION_MODEL,
        "VIDEO_SIZE": request.app.state.config.VIDEO_SIZE,
        "VIDEO_STEPS": request.app.state.config.VIDEO_STEPS,
    }


@router.post("/video/config/update")
async def update_video_config(
    request: Request, form_data: VideoConfigForm, user=Depends(get_admin_user)
):
    set_video_model(request, form_data.MODEL)

    pattern = r"^\d+x\d+$"
    if re.match(pattern, form_data.VIDEO_SIZE):
        request.app.state.config.VIDEO_SIZE = form_data.VIDEO_SIZE
    else:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("  (e.g., 512x512)."),
        )

    if form_data.VIDEO_STEPS >= 0:
        request.app.state.config.VIDEO_STEPS = form_data.VIDEO_STEPS
    else:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("  (e.g., 50)."),
        )

    return {
        "MODEL": request.app.state.config.VIDEO_GENERATION_MODEL,
        "VIDEO_SIZE": request.app.state.config.VIDEO_SIZE,
        "VIDEO_STEPS": request.app.state.config.VIDEO_STEPS,
    }


@router.get("/videos/models")
def get_models(request: Request, user=Depends(get_verified_user)):
    try:
        if request.app.state.config.VIDEO_GENERATION_ENGINE == "comfyui":
            # TODO - get models from comfyui
            headers = {
                "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
            }
            r = requests.get(
                url=f"{request.app.state.config.COMFYUI_BASE_URL}/object_info",
                headers=headers,
            )
            info = r.json()

            workflow = json.loads(request.app.state.config.COMFYUI_WORKFLOW)
            model_node_id = None

            for node in request.app.state.config.COMFYUI_WORKFLOW_NODES:
                if node["type"] == "model":
                    if node["node_ids"]:
                        model_node_id = node["node_ids"][0]
                    break

            if model_node_id:
                model_list_key = None

                log.info(workflow[model_node_id]["class_type"])
                for key in info[workflow[model_node_id]["class_type"]]["input"][
                    "required"
                ]:
                    if "_name" in key:
                        model_list_key = key
                        break

                if model_list_key:
                    return list(
                        map(
                            lambda model: {"id": model, "name": model},
                            info[workflow[model_node_id]["class_type"]]["input"][
                                "required"
                            ][model_list_key][0],
                        )
                    )
            else:
                return list(
                    map(
                        lambda model: {"id": model, "name": model},
                        info["CheckpointLoaderSimple"]["input"]["required"][
                            "ckpt_name"
                        ][0],
                    )
                )
    except Exception as e:
        request.app.state.config.ENABLE_VIDEO_GENERATION = False
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class GenerateVideoForm(BaseModel):
    model: Optional[str] = None
    prompt: str
    size: Optional[str] = None
    n: int = 1
    negative_prompt: Optional[str] = None


def load_b64_video_data(b64_str):
    try:
        if "," in b64_str:
            header, encoded = b64_str.split(",", 1)
            mime_type = header.split(";")[0]
            img_data = base64.b64decode(encoded)
        else:
            mime_type = "video/mp4"
            img_data = base64.b64decode(b64_str)
        return img_data, mime_type
    except Exception as e:
        log.exception(f"Error loading video data: {e}")
        return None


def load_url_video_data(url, headers=None):
    try:
        if headers:
            r = requests.get(url, headers=headers)
        else:
            r = requests.get(url)

        r.raise_for_status()
        if r.headers["content-type"].split("/")[0] == "video":
            mime_type = r.headers["content-type"]
            return r.content, mime_type
        else:
            log.error("Url does not point to an video.")
            return None

    except Exception as e:
        log.exception(f"Error saving video: {e}")
        return None


def upload_video(request, video_metadata, video_data, content_type, user):
    video_format = mimetypes.guess_extension(content_type)
    file = UploadFile(
        file=io.BytesIO(video_data),
        filename=f"generated-video{video_format}",  # will be converted to a unique ID on upload_file
        headers={
            "content-type": content_type,
        },
    )
    file_item = upload_file(request, file, user, file_metadata=video_metadata)
    url = request.app.url_path_for("get_file_content_by_id", id=file_item.id)
    return url


@router.post("/videos/generations")
async def video_generations(
    request: Request,
    form_data: GenerateVideoForm,
    user=Depends(get_verified_user),
):
    width, height = tuple(map(int, request.app.state.config.VIDEO_SIZE.split("x")))

    r = None
    try:
        if request.app.state.config.VIDEO_GENERATION_ENGINE == "comfyui":
            data = {
                "prompt": form_data.prompt,
                "width": width,
                "height": height,
                "n": form_data.n,
            }

            if request.app.state.config.VIDEO_STEPS is not None:
                data["steps"] = request.app.state.config.VIDEO_STEPS

            if form_data.negative_prompt is not None:
                data["negative_prompt"] = form_data.negative_prompt

            form_data = ComfyUIGenerateVideoForm(
                **{
                    "workflow": ComfyUIWorkflow(
                        **{
                            "workflow": request.app.state.config.COMFYUI_WORKFLOW,
                            "nodes": request.app.state.config.COMFYUI_WORKFLOW_NODES,
                        }
                    ),
                    **data,
                }
            )
            res = await comfyui_generate_video(
                request.app.state.config.VIDEO_GENERATION_MODEL,
                form_data,
                user.id,
                request.app.state.config.COMFYUI_BASE_URL,
                request.app.state.config.COMFYUI_API_KEY,
            )
            log.debug(f"res: {res}")

            videos = []

            for video in res["data"]:
                headers = None
                if request.app.state.config.COMFYUI_API_KEY:
                    headers = {
                        "Authorization": f"Bearer {request.app.state.config.COMFYUI_API_KEY}"
                    }

                video_data, content_type = load_url_video_data(video["url"], headers)
                url = upload_video(
                    request,
                    form_data.model_dump(exclude_none=True),
                    video_data,
                    content_type,
                    user,
                )
                videos.append({"url": url})
            return videos
    except Exception as e:
        error = e
        if r != None:
            data = r.json()
            if "error" in data:
                error = data["error"]["message"]
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(error))
