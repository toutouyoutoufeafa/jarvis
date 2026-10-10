"""Monitor Agent - System surveillance, log analysis, alerting."""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
from loguru import logger

from jarvis.agent.config import get_agent_config
from jarvis.agent.core import get_agent_core
from jarvis.agent.memory import get_memory_manager
from jarvis.agent.tools import get_tool_registry
from jarvis.llm import get_llm_engine
from jarvis.events import Event, get_event_bus


@dataclass
class MetricSnapshot:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: dict
    process_count: int
    top_processes: list[dict]


@dataclass
class Alert:
    id: str
    level: str  # info, warning, critical
    source: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    acknowledged: bool = False


class MonitorAgent:
    """Continuous system monitoring and alerting agent."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self.llm = get_llm_engine()
        self.memory = get_memory_manager()
        self.tools = get_tool_registry()
        self.core = get_agent_core()
        self.event_bus = get_event_bus()

        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._metrics_history: list[MetricSnapshot] = []
        self._alerts: list[Alert] = []
        self._log_positions: dict[str, int] = {}

    async def initialize(self) -> None:
        await self.core.initialize()
        logger.info("Monitor agent initialized")

    async def start_monitoring(self) -> None:
        """Start continuous monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Monitor agent started")

    async def stop_monitoring(self) -> None:
        """Stop monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitor agent stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                snapshot = await self._collect_metrics()
                self._metrics_history.append(snapshot)

                # Keep last 1000 snapshots
                if len(self._metrics_history) > 1000:
                    self._metrics_history = self._metrics_history[-1000:]

                # Check thresholds
                await self._check_thresholds(snapshot)

                # Analyze logs
                await self._analyze_logs()

                # Store in memory periodically
                if len(self._metrics_history) % 10 == 0:
                    await self._store_metrics_summary()

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            await asyncio.sleep(self.config.monitor.check_interval)

    async def _collect_metrics(self) -> MetricSnapshot:
        """Collect system metrics."""
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net = psutil.net_io_counters()

        # Top processes by CPU
        procs = []
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = p.info
                if info['cpu_percent'] and info['cpu_percent'] > 1.0:
                    procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        procs.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)

        return MetricSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu,
            memory_percent=mem.percent,
            disk_percent=disk.percent,
            network_io={"bytes_sent": net.bytes_sent, "bytes_recv": net.bytes_recv},
            process_count=len(list(psutil.process_iter())),
            top_processes=procs[:10],
        )

    async def _check_thresholds(self, snapshot: MetricSnapshot) -> None:
        """Check metrics against thresholds."""
        thresholds = self.config.monitor.alert_thresholds

        checks = [
            ("cpu_percent", snapshot.cpu_percent, "CPU usage"),
            ("memory_percent", snapshot.memory_percent, "Memory usage"),
            ("disk_percent", snapshot.disk_percent, "Disk usage"),
        ]

        for key, value, name in checks:
            threshold = thresholds.get(key, 90.0)
            if value >= threshold:
                await self._create_alert(
                    level="critical" if value >= threshold * 1.1 else "warning",
                    source="system_metrics",
                    message=f"{name} at {value:.1f}% (threshold: {threshold}%)",
                    metadata={"metric": key, "value": value, "threshold": threshold},
                )

    async def _analyze_logs(self) -> None:
        """Analyze log files for anomalies."""
        for log_path_str in self.config.monitor.log_paths:
            log_path = Path(log_path_str).expanduser()
            if not log_path.exists():
                continue

            try:
                # Read new lines since last check
                pos = self._log_positions.get(str(log_path), 0)
                with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                    f.seek(pos)
                    new_lines = f.readlines()
                    self._log_positions[str(log_path)] = f.tell()

                if not new_lines:
                    continue

                # Quick pattern matching for common issues
                error_patterns = [
                    (r"(?i)(error|exception|fail|critical|fatal)", "error"),
                    (r"(?i)(warn|warning)", "warning"),
                    (r"(?i)(timeout|connection refused|unreachable)", "connectivity"),
                    (r"(?i)(out of memory|oom|memory limit)", "memory"),
                    (r"(?i)(disk full|no space left)", "disk"),
                ]

                for line in new_lines[-100:]:  # Last 100 lines
                    for pattern, category in error_patterns:
                        if re.search(pattern, line):
                            await self._create_alert(
                                level="warning",
                                source=f"log:{log_path.name}",
                                message=f"Log anomaly detected: {line.strip()[:200]}",
                                metadata={"category": category, "log_file": str(log_path)},
                            )
                            break

            except Exception as e:
                logger.debug(f"Log analysis error for {log_path}: {e}")

    async def _create_alert(self, level: str, source: str, message: str, metadata: dict) -> None:
        """Create and store alert."""
        alert = Alert(
            id=f"alert_{int(time.time() * 1000)}",
            level=level,
            source=source,
            message=message,
            metadata=metadata,
        )

        self._alerts.append(alert)
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

        # Store in memory
        self.memory.add_stm(
            f"ALERT [{level.upper()}] {source}: {message}",
            metadata={"type": "alert", "level": level, **metadata},
        )

        # Emit event
        await self.event_bus.emit(Event("monitor.alert", {
            "alert": {
                "id": alert.id,
                "level": alert.level,
                "source": alert.source,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }
        }))

        logger.warning(f"ALERT [{level}] {source}: {message}")

    async def _store_metrics_summary(self) -> None:
        """Store metrics summary in long-term memory."""
        if not self._metrics_history:
            return

        recent = self._metrics_history[-10:]
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_mem = sum(m.memory_percent for m in recent) / len(recent)
        avg_disk = sum(m.disk_percent for m in recent) / len(recent)

        summary = f"System metrics (last {len(recent)} samples): CPU={avg_cpu:.1f}%, RAM={avg_mem:.1f}%, Disk={avg_disk:.1f}%"

        self.memory.add_stm(summary, metadata={"type": "metrics_summary"})

    async def get_status(self) -> dict:
        """Get current monitoring status."""
        latest = self._metrics_history[-1] if self._metrics_history else None

        return {
            "running": self._running,
            "current_metrics": {
                "cpu_percent": latest.cpu_percent if latest else None,
                "memory_percent": latest.memory_percent if latest else None,
                "disk_percent": latest.disk_percent if latest else None,
                "process_count": latest.process_count if latest else None,
            } if latest else None,
            "recent_alerts": [
                {
                    "id": a.id,
                    "level": a.level,
                    "source": a.source,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self._alerts[-10:]
            ],
            "alert_counts": {
                "critical": sum(1 for a in self._alerts if a.level == "critical"),
                "warning": sum(1 for a in self._alerts if a.level == "warning"),
                "info": sum(1 for a in self._alerts if a.level == "info"),
            },
        }

    async def get_metrics_history(self, minutes: int = 60) -> list[dict]:
        """Get metrics history for specified minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "disk_percent": m.disk_percent,
            }
            for m in self._metrics_history
            if m.timestamp >= cutoff
        ]

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    async def run_health_check(self) -> dict:
        """Run comprehensive health check."""
        checks = {}

        # System resources
        snapshot = await self._collect_metrics()
        checks["system"] = {
            "cpu_ok": snapshot.cpu_percent < 80,
            "memory_ok": snapshot.memory_percent < 80,
            "disk_ok": snapshot.disk_percent < 85,
        }

        # Check key services (customize as needed)
        services = ["docker", "ollama", "postgresql", "redis"]
        for svc in services:
            try:
                result = await self.tools.execute("shell_exec", command=f"systemctl is-active {svc}")
                checks[f"service_{svc}"] = result.get("success", False)
            except Exception:
                checks[f"service_{svc}"] = False

        # Overall
        all_ok = all(v.get("cpu_ok", True) and v.get("memory_ok", True) and v.get("disk_ok", True)
                     for k, v in checks.items() if isinstance(v, dict))

        return {
            "healthy": all_ok,
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }


# Global instance
_monitor: MonitorAgent | None = None


def get_monitor_agent() -> MonitorAgent:
    global _monitor
    if _monitor is None:
        _monitor = MonitorAgent()
    return _monitor