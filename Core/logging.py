"""统一日志系统 — 每次程序启动创建独立的日志 session。

日志目录结构::

    Logs/
    └── {script_name}/
        └── {YYYYMMDD}_{HHMMSS}/
            ├── run.log          # 完整日志 (file_level)
            └── error.log        # 仅 WARNING+ 级别

用法::

    from Core.logging import setup_run_logger

    log_dir = setup_run_logger(__file__)            # 用默认级别
    log_dir = setup_run_logger(__file__, "config.yaml")  # 从 YAML 读取级别
    log = logging.getLogger(__name__)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

# 项目根目录 (Core/ 的父目录)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════
#  彩色终端 Formatter
# ══════════════════════════════════════════════════════════════

# ANSI 颜色代码
_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """为终端输出添加颜色。仅在 stream 为 tty 时着色。"""

    def __init__(self, fmt: str, datefmt: str | None = None, stream: TextIO | None = None) -> None:
        super().__init__(fmt, datefmt)
        self._use_color = (stream or sys.stderr).isatty()

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._use_color:
            color = _COLORS.get(record.levelname, "")
            if color:
                # 只给级别名着色
                msg = msg.replace(
                    f"[{record.levelname}]",
                    f"{color}[{record.levelname}]{_RESET}",
                    1,
                )
        return msg


# ══════════════════════════════════════════════════════════════
#  核心 API
# ══════════════════════════════════════════════════════════════

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_level(level: str | int) -> int:
    """将字符串级别名或整数转为 logging 级别常量。"""
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper(), logging.INFO)


def setup_run_logger(
    script_path: str,
    config_path: str | Path | None = None,
    *,
    log_root: str | Path = "Logs",
    console_level: int | str | None = None,
    file_level: int | str | None = None,
) -> Path:
    """为本次运行配置 root logger，返回日志目录路径。

    Args:
        script_path: 调用脚本的 ``__file__``，用于提取脚本名。
        config_path: YAML 配置文件路径。若提供且包含 ``logging`` 段，
            则从中读取 ``console_level`` 和 ``file_level``。
        log_root: 日志根目录 (相对于项目根目录或绝对路径)。
        console_level: 终端输出级别，覆盖 YAML 配置。默认 INFO。
        file_level: 文件输出级别，覆盖 YAML 配置。默认 DEBUG。

    Returns:
        本次运行的日志目录 Path。
    """
    # ── 从 YAML 读取日志配置 ──
    yaml_console_level = None
    yaml_file_level = None
    if config_path is not None:
        yaml_console_level, yaml_file_level = _read_yaml_log_config(config_path)

    # 优先级: 函数参数 > YAML 配置 > 默认值
    resolved_console = _parse_level(
        console_level if console_level is not None
        else yaml_console_level if yaml_console_level is not None
        else logging.INFO
    )
    resolved_file = _parse_level(
        file_level if file_level is not None
        else yaml_file_level if yaml_file_level is not None
        else logging.DEBUG
    )

    # ── 创建日志目录 ──
    script_name = Path(script_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_root_path = Path(log_root)
    if not log_root_path.is_absolute():
        log_root_path = _PROJECT_ROOT / log_root_path

    log_dir = log_root_path / script_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── 配置 root logger ──
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root 设最低，由 handler 控制过滤

    # 清除已有 handler (避免重复调用时叠加)
    root.handlers.clear()

    # 1. 终端 handler (彩色)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(resolved_console)
    console_handler.setFormatter(ColorFormatter(_LOG_FORMAT, _DATE_FORMAT, stream=sys.stdout))
    root.addHandler(console_handler)

    # 2. 全量日志文件
    run_log_path = log_dir / "run.log"
    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setLevel(resolved_file)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(file_handler)

    # 3. 错误日志文件 (WARNING+)
    error_log_path = log_dir / "error.log"
    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(error_handler)

    # ── 启动日志 ──
    logger = logging.getLogger(__name__)
    logger.info("日志目录: %s", log_dir)
    logger.debug("console_level=%s, file_level=%s",
                 logging.getLevelName(resolved_console),
                 logging.getLevelName(resolved_file))

    return log_dir


def _read_yaml_log_config(config_path: str | Path) -> tuple[str | None, str | None]:
    """从 YAML 配置文件读取 logging 段的级别设置。

    YAML 格式::

        logging:
          console_level: INFO    # 终端输出级别
          file_level: DEBUG      # 文件输出级别

    Returns:
        (console_level, file_level) 字符串或 None。
    """
    try:
        import yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        log_cfg = raw.get("logging")
        if isinstance(log_cfg, dict):
            return log_cfg.get("console_level"), log_cfg.get("file_level")
    except Exception as e:
        import sys
        print(f"[WARNING] 读取日志配置失败 ({config_path}): {e}", file=sys.stderr)
    return None, None
