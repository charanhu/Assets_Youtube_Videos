import inspect
import logging
import os
from base64 import b64encode
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
from IPython.display import display
from pydantic import BaseModel
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vision_agent.tools.tools_types import BoundingBoxes
from vision_agent.utils.exceptions import RemoteToolCallFailed
from vision_agent.utils.execute import Error, MimeType
from vision_agent.utils.image_utils import normalize_bbox
from vision_agent.utils.type_defs import LandingaiAPIKey

_LOGGER = logging.getLogger(__name__)
_LND_API_KEY = os.environ.get("LANDINGAI_API_KEY", LandingaiAPIKey().api_key)
_LND_BASE_URL = os.environ.get("LANDINGAI_URL", "https://api.landing.ai")
_LND_API_URL = f"{_LND_BASE_URL}/v1/agent/model"
_LND_API_URL_v2 = f"{_LND_BASE_URL}/v1/tools"


class ToolCallTrace(BaseModel):
    endpoint_url: str
    type: str
    request: MutableMapping[str, Any]
    response: MutableMapping[str, Any]
    error: Optional[Error]
    files: Optional[List[tuple[str, str]]]


def send_inference_request(
    payload: Dict[str, Any],
    endpoint_name: str,
    files: Optional[List[Tuple[Any, ...]]] = None,
    v2: bool = False,
    metadata_payload: Optional[Dict[str, Any]] = None,
    is_form: bool = False,
) -> Any:
    url = f"{_LND_API_URL_v2 if v2 else _LND_API_URL}/{endpoint_name}"
    if "TOOL_ENDPOINT_URL" in os.environ:
        url = os.environ["TOOL_ENDPOINT_URL"]

    headers = {"apikey": _LND_API_KEY}
    if "TOOL_ENDPOINT_AUTH" in os.environ:
        headers["Authorization"] = os.environ["TOOL_ENDPOINT_AUTH"]
        headers.pop("apikey")

    if runtime_tag := os.environ.get("RUNTIME_TAG", ""):
        headers["runtime_tag"] = runtime_tag

    session = _create_requests_session(
        url=url,
        num_retry=3,
        headers=headers,
    )

    function_name = "unknown"
    if "function_name" in payload:
        function_name = payload["function_name"]
    elif metadata_payload is not None and "function_name" in metadata_payload:
        function_name = metadata_payload["function_name"]

    response = _call_post(url, payload, session, files, function_name, is_form)

    # TODO: consider making the response schema the same between below two sources
    return response if "TOOL_ENDPOINT_AUTH" in os.environ else response["data"]


def send_task_inference_request(
    payload: Dict[str, Any],
    task_name: str,
    files: Optional[List[Tuple[Any, ...]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    is_form: bool = False,
) -> Any:
    url = f"{_LND_API_URL_v2}/{task_name}"
    headers = {"apikey": _LND_API_KEY}
    session = _create_requests_session(
        url=url,
        num_retry=3,
        headers=headers,
    )

    function_name = "unknown"
    if metadata is not None and "function_name" in metadata:
        function_name = metadata["function_name"]
    response = _call_post(url, payload, session, files, function_name, is_form)
    return response["data"]


def _create_requests_session(
    url: str, num_retry: int, headers: Dict[str, str]
) -> Session:
    """Create a requests session with retry"""
    session = Session()
    retries = Retry(
        total=num_retry,
        backoff_factor=2,
        raise_on_redirect=True,
        raise_on_status=False,
        allowed_methods=["GET", "POST", "PUT"],
        status_forcelist=[
            408,  # Request Timeout
            429,  # Too Many Requests (ie. rate limiter).
            502,  # Bad Gateway
            503,  # Service Unavailable (include cloud circuit breaker)
            504,  # Gateway Timeout
        ],
    )
    session.mount(url, HTTPAdapter(max_retries=retries if num_retry > 0 else 0))
    session.headers.update(headers)
    return session


def get_tool_documentation(funcs: List[Callable[..., Any]]) -> str:
    docstrings = ""
    for func in funcs:
        docstrings += f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}\n\n"

    return docstrings


def get_tool_descriptions(funcs: List[Callable[..., Any]]) -> str:
    descriptions = ""
    for func in funcs:
        description = func.__doc__
        if description is None:
            description = ""

        if "Parameters:" in description:
            description = (
                description[: description.find("Parameters:")]
                .replace("\n", " ")
                .strip()
            )

        description = " ".join(description.split())
        descriptions += f"- {func.__name__}{inspect.signature(func)}: {description}\n"
    return descriptions


def get_tool_descriptions_by_names(
    tool_name: Optional[List[str]],
    funcs: List[Callable[..., Any]],
    util_funcs: List[
        Callable[..., Any]
    ],  # util_funcs will always be added to the list of functions
) -> str:
    if tool_name is None:
        return get_tool_descriptions(funcs + util_funcs)

    invalid_names = [
        name for name in tool_name if name not in {func.__name__ for func in funcs}
    ]

    if invalid_names:
        raise ValueError(f"Invalid customized tool names: {', '.join(invalid_names)}")

    filtered_funcs = (
        funcs
        if not tool_name
        else [func for func in funcs if func.__name__ in tool_name]
    )
    return get_tool_descriptions(filtered_funcs + util_funcs)


def get_tools_df(funcs: List[Callable[..., Any]]) -> pd.DataFrame:
    data: Dict[str, List[str]] = {"desc": [], "doc": [], "name": []}

    for func in funcs:
        desc = func.__doc__
        if desc is None:
            desc = ""
        desc = desc[: desc.find("Parameters:")].replace("\n", " ").strip()
        desc = " ".join(desc.split())

        doc = f"{func.__name__}{inspect.signature(func)}:\n{func.__doc__}"
        data["desc"].append(desc)
        data["doc"].append(doc)
        data["name"].append(func.__name__)

    return pd.DataFrame(data)  # type: ignore


def get_tools_info(funcs: List[Callable[..., Any]]) -> Dict[str, str]:
    data: Dict[str, str] = {}

    for func in funcs:
        desc = func.__doc__
        if desc is None:
            desc = ""

        data[func.__name__] = f"{func.__name__}{inspect.signature(func)}:\n{desc}"

    return data


def _call_post(
    url: str,
    payload: dict[str, Any],
    session: Session,
    files: Optional[List[Tuple[Any, ...]]] = None,
    function_name: str = "unknown",
    is_form: bool = False,
) -> Any:
    files_in_b64 = None
    if files:
        files_in_b64 = [(file[0], b64encode(file[1]).decode("utf-8")) for file in files]

    tool_call_trace = None
    try:
        if files is not None:
            response = session.post(url, data=payload, files=files)
        elif is_form:
            response = session.post(url, data=payload)
        else:
            response = session.post(url, json=payload)

        tool_call_trace_payload = (
            payload
            if "function_name" in payload
            else {**payload, **{"function_name": function_name}}
        )
        tool_call_trace = ToolCallTrace(
            endpoint_url=url,
            type="tool_call",
            request=tool_call_trace_payload,
            response={},
            error=None,
            files=files_in_b64,
        )

        if response.status_code != 200:
            tool_call_trace.error = Error(
                name="RemoteToolCallFailed",
                value=f"{response.status_code} - {response.text}",
                traceback_raw=[],
            )
            _LOGGER.error(f"Request failed: {response.status_code} {response.text}")
            raise RemoteToolCallFailed(
                function_name, response.status_code, response.text
            )

        result = response.json()
        tool_call_trace.response = result
        return result
    finally:
        if tool_call_trace is not None:
            trace = tool_call_trace.model_dump()
            display({MimeType.APPLICATION_JSON: trace}, raw=True)


def filter_bboxes_by_threshold(
    bboxes: BoundingBoxes, threshold: float
) -> BoundingBoxes:
    return list(filter(lambda bbox: bbox.score >= threshold, bboxes))


def add_bboxes_from_masks(
    all_preds: List[List[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    for frame_preds in all_preds:
        for preds in frame_preds:
            if np.sum(preds["mask"]) == 0:
                preds["bbox"] = []
            else:
                rows, cols = np.where(preds["mask"])
                bbox = [
                    float(np.min(cols)),
                    float(np.min(rows)),
                    float(np.max(cols)),
                    float(np.max(rows)),
                ]
                bbox = normalize_bbox(bbox, preds["mask"].shape)
                preds["bbox"] = bbox

    return all_preds


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    intersection = x_overlap * y_overlap

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def single_nms(
    preds: List[Dict[str, Any]], iou_threshold: float
) -> List[Dict[str, Any]]:
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if calculate_iou(preds[i]["bbox"], preds[j]["bbox"]) > iou_threshold:
                if preds[i]["score"] > preds[j]["score"]:
                    preds[j]["score"] = 0
                else:
                    preds[i]["score"] = 0

    return [pred for pred in preds if pred["score"] > 0]


def nms(
    all_preds: List[List[Dict[str, Any]]], iou_threshold: float
) -> List[List[Dict[str, Any]]]:
    return_preds = []
    for frame_preds in all_preds:
        frame_preds = single_nms(frame_preds, iou_threshold)
        return_preds.append(frame_preds)

    return return_preds