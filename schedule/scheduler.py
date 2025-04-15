from abc import ABC, abstractmethod
from typing import List, Dict


class ScheduleSLAChecker:
    """A class to check if the SLA is met for a given request."""

    def __init__(self, config: Dict) -> None:
        self.config = config

    def check_sla(self, request: Dict) -> bool:
        """Check if the SLA is met for the given request."""
        pass


class Scheduler(ABC):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self._optimal_target = config.get("optimal_target", "latency")
        self._sla_checker = ScheduleSLAChecker(config)

    @abstractmethod
    def schedule(self, request: List[Dict]) -> int:
        """Schedule the request."""
        pass
