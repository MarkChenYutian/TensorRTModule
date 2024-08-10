import logging
import os
from typing import Any, Literal
from rich.console import Console
from rich.logging import RichHandler

IS_TERMINAL_MODE = False
TERMINAL_WIDTH = 150
try:
    TERMINAL_WIDTH = os.get_terminal_size().columns
    IS_TERMINAL_MODE = True
except OSError:
    pass
IS_TERMINAL_MODE = True  # For debugging
GlobalConsole = Console(width=TERMINAL_WIDTH)


class GlobalLog:
    LogLevel = Literal["info", "error", "warn", "fatal"]
    
    LOCK: "None | GlobalLog" = None
    Translate = {
        "info": logging.INFO,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "fatal": logging.FATAL,
    }
    
    def __new__(cls, *args, **kwargs) -> "GlobalLog":
        if GlobalLog.LOCK is not None: return GlobalLog.LOCK
        return super().__new__(cls)

    def __init__(self) -> None:
        if GlobalLog.LOCK: return
        GlobalLog.LOCK = self
        
        logging.getLogger("evo").setLevel(logging.CRITICAL)
        logging.getLogger("timm").setLevel(logging.CRITICAL)
        logging.getLogger("numexpr").setLevel(logging.CRITICAL)
        
        logging.basicConfig(
            level="INFO",
            format="PID %(process)d %(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=GlobalConsole)],
        )
        self.__logger = logging.getLogger("AirVIO")

    def write(self, level: LogLevel, msg: Any, marked: bool = False) -> None:
        lg_level = self.Translate[level]
        self.__logger.log(lg_level, msg, stacklevel=2, extra={"markup": marked})

Logger = GlobalLog()
