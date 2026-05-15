from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from gym_torcs.constants import DEFAULT_TORCS_EXECUTABLE, SCR_MAX_PORT, SCR_MIN_PORT


def scr_idx_from_port(port: int) -> int:
    """Map SCR UDP ports to scr_server driver indices.

    Standard SCR server robots expose ports 3001..3010 through idx 0..9.
    """
    port = int(port)
    if not (SCR_MIN_PORT <= port <= SCR_MAX_PORT):
        raise ValueError(
            f"SCR TORCS supports ports {SCR_MIN_PORT}..{SCR_MAX_PORT} via "
            f"scr_server idx 0..9. Got port={port}."
        )
    return port - SCR_MIN_PORT


@dataclass(slots=True)
class RaceConfig:
    """Complete TORCS race setup controlled by Python, not by user XML editing."""

    race_type: str = "practice"
    track_name: str = "michigan"
    track_category: str = "oval"
    laps: int = 20
    scr_idx: int = 0

    @property
    def race_file_name(self) -> str:
        race_type = self.race_type.lower().strip()
        if race_type != "practice":
            raise ValueError(
                "Only race_type='practice' is currently supported by the Gym wrapper. "
                f"Got race_type={self.race_type!r}."
            )
        return "practice.xml"

    @property
    def display_name(self) -> str:
        return "Practice"


class RaceXml:
    """Writes TORCS raceman XML from RaceConfig.

    Users should normally pass race_type/track_name/track_category/laps/port to
    TorcsEnv.  A template can still be passed for advanced installs, but it is
    no longer part of the normal public setup.
    """

    def __init__(self, template: str | Path | None = None) -> None:
        self.template = Path(template).expanduser() if template else None

    def write(self, destination: str | Path, cfg: RaceConfig) -> None:
        destination = Path(destination).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        if self.template is not None:
            tree = ET.parse(self.template)
            root = tree.getroot()
        else:
            root = self._default_root(cfg)
            tree = ET.ElementTree(root)

        self._patch(root, cfg)
        self._indent(root)
        tree.write(destination, encoding="UTF-8", xml_declaration=True)

    def _patch(self, root: ET.Element, cfg: RaceConfig) -> None:
        # Header / race identity
        root.set("name", cfg.display_name)
        header = self._section(root, "./section[@name='Header']")
        self._set_node(header, "name", cfg.display_name, "attstr")
        self._set_node(header, "description", cfg.display_name, "attstr")
        self._set_node(header, "priority", "100", "attnum")
        self._set_node(header, "menu image", "data/img/splash-practice.png", "attstr")
        self._set_node(header, "run image", "data/img/splash-run-practice.png", "attstr")

        # Track selection
        tracks = self._section(root, "./section[@name='Tracks']")
        self._set_node(tracks, "maximum number", "1", "attnum")
        track = self._section(tracks, "./section[@name='1']")
        self._set_node(track, "name", cfg.track_name, "attstr")
        self._set_node(track, "category", cfg.track_category, "attstr")

        # Race list and practice settings
        races = self._section(root, "./section[@name='Races']")
        race = self._section(races, "./section[@name='1']")
        self._set_node(race, "name", cfg.display_name, "attstr")

        practice = self._section(root, "./section[@name='Practice']")
        self._set_node(practice, "laps", str(int(cfg.laps)), "attnum")
        self._set_node(practice, "type", "practice", "attstr")
        self._set_node(practice, "starting order", "drivers list", "attstr")
        self._set_node(practice, "restart", "yes", "attstr")
        self._set_node(practice, "display mode", "normal", "attstr")
        self._set_node(practice, "display results", "no", "attstr")
        self._set_node(practice, "distance", "0", "attnum", unit="km")

        grid = self._section(practice, "./section[@name='Starting Grid']")
        self._set_node(grid, "rows", "1", "attnum")
        self._set_node(grid, "distance to start", "100", "attnum")
        self._set_node(grid, "distance between columns", "20", "attnum")
        self._set_node(grid, "offset within a column", "10", "attnum")
        self._set_node(grid, "initial speed", "0", "attnum", unit="km/h")
        self._set_node(grid, "initial height", "0.2", "attnum", unit="m")

        # SCR driver / UDP port selection
        idx = str(int(cfg.scr_idx))
        drivers = self._section(root, "./section[@name='Drivers']")
        self._set_node(drivers, "maximum number", "1", "attnum")
        self._set_node(drivers, "focused module", "scr_server", "attstr")
        self._set_node(drivers, "focused idx", idx, "attnum")
        driver = self._section(drivers, "./section[@name='1']")
        self._set_node(driver, "module", "scr_server", "attstr")
        self._set_node(driver, "idx", idx, "attnum")

        start_list = self._section(root, "./section[@name='Drivers Start List']")
        start = self._section(start_list, "./section[@name='1']")
        self._set_node(start, "module", "scr_server", "attstr")
        self._set_node(start, "idx", idx, "attnum")

        # Make TORCS frontend land on a complete Practice configuration.
        config = self._section(root, "./section[@name='Configuration']")
        self._set_node(config, "current configuration", "4", "attnum")
        c1 = self._section(config, "./section[@name='1']")
        self._set_node(c1, "type", "track select", "attstr")
        c2 = self._section(config, "./section[@name='2']")
        self._set_node(c2, "type", "drivers select", "attstr")
        c3 = self._section(config, "./section[@name='3']")
        self._set_node(c3, "type", "race config", "attstr")
        self._set_node(c3, "race", cfg.display_name, "attstr")
        opts = self._section(c3, "./section[@name='Options']")
        o1 = self._section(opts, "./section[@name='1']")
        self._set_node(o1, "type", "race length", "attstr")
        o2 = self._section(opts, "./section[@name='2']")
        self._set_node(o2, "type", "display mode", "attstr")

    @staticmethod
    def _default_root(cfg: RaceConfig) -> ET.Element:
        return ET.Element("params", {"name": cfg.display_name})

    @staticmethod
    def _section(parent: ET.Element, path: str) -> ET.Element:
        found = parent.find(path)
        if found is not None:
            return found
        if not path.startswith("./section[@name='"):
            raise ValueError(f"Unsupported XML creation path: {path!r}")
        name = path.split("./section[@name='")[1].split("']")[0]
        return ET.SubElement(parent, "section", {"name": name})

    @staticmethod
    def _set_node(section: ET.Element, name: str, value: str, kind: str, *, unit: str | None = None) -> None:
        node = section.find(f"./{kind}[@name='{name}']")
        if node is None:
            node = ET.SubElement(section, kind, {"name": name})
        node.set("val", value)
        if unit is not None:
            node.set("unit", unit)

    @staticmethod
    def _indent(elem: ET.Element, level: int = 0) -> None:
        pad = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = pad + "  "
            for child in elem:
                RaceXml._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


class TorcsServer:
    """Owns exactly one TORCS process and writes its race setup."""

    def __init__(
        self,
        *,
        port: int,
        render_mode: str | None,
        executable: str = DEFAULT_TORCS_EXECUTABLE,
        template_xml: str | Path | None = None,
        source_torcs_home: str | Path | None = None,
        workdir: str | Path | None = None,
        race_config: RaceConfig | None = None,
        startup_sleep: float = 2.0,
        debug: bool = False,
        gui_auto_start: bool = True,
        gui_start_delay: float = 2.0,
        gui_key_delay: float = 0.10,
        gui_auto_start_keys: tuple[str, ...] = ("Return", "Return", "Up", "Up", "Return", "Return"),
    ) -> None:
        self.port = int(port)
        self.scr_idx = scr_idx_from_port(self.port)
        self.render_mode = render_mode
        self.executable = str(Path(executable).expanduser())
        self.template_xml = Path(template_xml).expanduser() if template_xml else None
        self.source_torcs_home = Path(source_torcs_home).expanduser() if source_torcs_home else Path.home() / ".torcs"
        self.race_config = race_config or RaceConfig(scr_idx=self.scr_idx)
        self.race_config.scr_idx = self.scr_idx
        self.startup_sleep = startup_sleep
        self.debug = debug
        self.gui_auto_start = gui_auto_start
        self.gui_start_delay = gui_start_delay
        self.gui_key_delay = gui_key_delay
        self.gui_auto_start_keys = gui_auto_start_keys
        self.process: subprocess.Popen[bytes] | None = None

        if render_mode == "human":
            self.home = Path.home()
            self.owns_home = False
        else:
            self.home = Path(workdir).expanduser() if workdir else Path(tempfile.mkdtemp(prefix=f"torcs-{self.port}-"))
            self.owns_home = workdir is None

    @property
    def race_file(self) -> Path:
        return self.home / ".torcs" / "config" / "raceman" / self.race_config.race_file_name

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.running:
            return
        self.race_config.scr_idx = self.scr_idx
        self._prepare_config()
        cmd = self.command()
        env = self._env()
        if self.debug:
            print("[gym_torcs] version: multiport_pflag_v1", flush=True)
            print("[gym_torcs] command:", " ".join(cmd), flush=True)
            print("[gym_torcs] HOME:", env.get("HOME", str(Path.home())), flush=True)
            print("[gym_torcs] race_file:", self.race_file, flush=True)
            print("[gym_torcs] race_type:", self.race_config.race_type, flush=True)
            print("[gym_torcs] track:", f"{self.race_config.track_category}/{self.race_config.track_name}", flush=True)
            print("[gym_torcs] port:", self.port, "scr_idx:", self.scr_idx, flush=True)
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=None if self.debug else subprocess.DEVNULL,
            stderr=None if self.debug else subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(self.startup_sleep)
        if self.render_mode == "human" and self.gui_auto_start:
            self._auto_start_gui_race()

    def restart(self, race_config: RaceConfig | None = None) -> None:
        if race_config is not None:
            self.race_config = race_config
        self.race_config.scr_idx = self.scr_idx
        self.stop()
        self.start()

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=5)
        self.process = None

    def close(self) -> None:
        self.stop()
        if self.owns_home:
            shutil.rmtree(self.home, ignore_errors=True)

    def command(self) -> list[str]:
        if self.render_mode == "human":
            # On this TORCS install, only `torcs` without race flags reliably
            # opens the GUI. The generated XML + gui_auto_start picks the race.
            return [self.executable]
        return [
            self.executable,
            "-T",
            "-nofuel",
            #"-nodamage",
            #"-nolaptime",
            "-p",
            str(self.port),
            "-r",
            str(self.race_file),
        ]

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.render_mode is None:
            env["HOME"] = str(self.home)
        return env

    def _auto_start_gui_race(self) -> None:
        time.sleep(self.gui_start_delay)
        tool = self._gui_key_tool()
        if tool is None:
            raise RuntimeError(
                "render_mode='human' needs a GUI key tool to start the TORCS race. "
                "Install one: sudo dnf install xautomation  # xte, or sudo dnf install xdotool."
            )
        if self.debug:
            print("[gym_torcs] gui_key_tool:", tool, flush=True)
            print("[gym_torcs] gui_auto_start_keys:", " ".join(self.gui_auto_start_keys), flush=True)

        xdotool = shutil.which("xdotool")
        if xdotool is not None:
            for query in (("--class", "torcs"), ("--name", "TORCS"), ("--name", "torcs")):
                result = subprocess.run(
                    [xdotool, "search", "--onlyvisible", *query],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                window_id = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
                if window_id:
                    subprocess.run([xdotool, "windowactivate", "--sync", window_id], check=False)
                    break

        for key in self.gui_auto_start_keys:
            if tool == "xte":
                subprocess.run(["xte", f"key {key}"], check=True)
            else:
                subprocess.run(["xdotool", "key", key], check=True)
            time.sleep(self.gui_key_delay)

    @staticmethod
    def _gui_key_tool() -> str | None:
        if shutil.which("xte") is not None:
            return "xte"
        if shutil.which("xdotool") is not None:
            return "xdotool"
        return None

    def _prepare_config(self) -> None:
        torcs_home = self.home / ".torcs"
        if torcs_home.resolve() != self.source_torcs_home.resolve():
            if not torcs_home.exists() and self.source_torcs_home.exists():
                shutil.copytree(self.source_torcs_home, torcs_home, dirs_exist_ok=True)
        RaceXml(self.template_xml).write(self.race_file, self.race_config)
