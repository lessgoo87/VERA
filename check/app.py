import webview
from bridge.api import VeraAPI

if __name__ == "__main__":
    api = VeraAPI()
    webview.create_window(
        title="VERA — Verification & Evidence Studio",
        url="index.html",
        js_api=api,
        width=1450,
        height=880
    )
    webview.start()