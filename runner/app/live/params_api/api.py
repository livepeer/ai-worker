import hashlib
import logging
import mimetypes
import os
import tempfile
import time
from typing import cast, Callable

from aiohttp import BodyPartReader, web
from asyncio import Awaitable

TEMP_SUBDIR = "infer_temp"
MAX_FILE_AGE = 86400  # 1 day


def cleanup_old_files(temp_dir):
    current_time = time.time()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > MAX_FILE_AGE:
                os.remove(file_path)
                logging.info(f"Removed old file: {file_path}")


async def handle_params_update(request):
    try:
        params = {}
        temp_dir = os.path.join(tempfile.gettempdir(), TEMP_SUBDIR)
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_old_files(temp_dir)

        if request.content_type.startswith("application/json"):
            params = await request.json()
        elif request.content_type.startswith("multipart/"):
            reader = await request.multipart()

            async for part in reader:
                if part.name == "params":
                    params.update(await part.json())
                elif isinstance(part, BodyPartReader):
                    content = await part.read()

                    file_hash = hashlib.md5(content).hexdigest()
                    content_type = part.headers.get(
                        "Content-Type", "application/octet-stream"
                    )
                    ext = mimetypes.guess_extension(content_type) or ""
                    new_filename = f"{file_hash}{ext}"

                    file_path = os.path.join(temp_dir, new_filename)
                    with open(file_path, "wb") as f:
                        f.write(content)

                    params[part.name] = file_path
        else:
            raise ValueError(f"Unknown content type: {request.content_type}")

        update_params = cast(Callable[[dict], Awaitable[None]], request.app["update_params_func"])
        await update_params(params)

        return web.Response(text="Params updated successfully")
    except Exception as e:
        logging.error(f"Error updating params: {e}")
        return web.Response(text=f"Error updating params: {str(e)}", status=400)


async def start_http_server(port: int, update_params: Callable[[dict], Awaitable[None]]):
    app = web.Application()
    app["update_params_func"] = update_params
    app.router.add_post("/api/params", handle_params_update)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"HTTP server started on port {port}")
    return runner
