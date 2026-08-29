import logging
import json
import sys
from datetime import datetime, timezone

class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        # Inject any custom telemetry dictionary passed in the 'extra' kwargs
        if hasattr(record, "telemetry"):
            log_record.update(record.telemetry)
            
        return json.dumps(log_record, ensure_ascii=False) # Ensure_ascii=False to ensure German umlauts could be printed

def setup_logger(name="KauflandVoiceBot"):
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONLogFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    # Suppress third-party library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logger