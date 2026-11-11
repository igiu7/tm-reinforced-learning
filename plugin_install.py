"""
Kopiuje wtyczkę tmi_plugin/RLBridge.as do folderu wtyczek TMInterface 2
(Dokumenty\\TMInterface\\Plugins), jeśli jej tam nie ma albo jest inna niż w projekcie.
Wywoływane automatycznie przy starcie menu; można też ręcznie:

    python plugin_install.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PLUGIN_SRC = Path(__file__).resolve().parent / "tmi_plugin" / "RLBridge.as"


def documents_dir() -> Path:
    """Folder Dokumenty - także gdy jest przeniesiony (np. do OneDrive)."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PERSONAL = 5 (Dokumenty), SHGFP_TYPE_CURRENT = 0
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:
                return Path(buf.value)
        except (OSError, AttributeError):
            pass
    return Path.home() / "Documents"


def plugins_dir(override: str | None = None) -> Path:
    return Path(override) if override else documents_dir() / "TMInterface" / "Plugins"


def install_plugin(override_dir: str | None = None) -> bool:
    """Zwraca True, gdy wtyczka w grze jest aktualna (po ewentualnym skopiowaniu)."""
    dest_dir = plugins_dir(override_dir)
    dest = dest_dir / PLUGIN_SRC.name

    if not PLUGIN_SRC.exists():
        print(f"[WTYCZKA] Brak pliku źródłowego {PLUGIN_SRC}")
        return False
    if not dest_dir.parent.exists() and override_dir is None:
        print(f"[WTYCZKA] Nie znaleziono folderu TMInterface ({dest_dir.parent}).\n"
              "          Zainstaluj TMInterface 2 i uruchom grę raz, albo ustaw TMI_PLUGINS_DIR w ql_config.py.")
        return False

    if dest.exists() and dest.read_bytes() == PLUGIN_SRC.read_bytes():
        print(f"[WTYCZKA] RL Bridge aktualna ({dest})")
        return True

    action = "Zaktualizowano" if dest.exists() else "Zainstalowano"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PLUGIN_SRC, dest)
    except OSError as e:
        print(f"[WTYCZKA] Nie udało się skopiować do {dest}: {e}")
        return False
    print(f"[WTYCZKA] {action} RL Bridge -> {dest}\n"
          "          Jeśli gra już działa, TMInterface przeładuje wtyczkę sam; jeśli nie - "
          "włącz ją w Settings > Plugins (albo komenda 'reload_plugin').")
    return True


if __name__ == "__main__":
    from ql_config import TMI_PLUGINS_DIR

    install_plugin(TMI_PLUGINS_DIR)
