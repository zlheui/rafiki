import os

from rafiki.utils.log import configure_logging
from rafiki.drift_detector.app import app

configure_logging('drift_detector')

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=os.getenv('DRIFT_DETECTOR_PORT', 8005), 
        debug=True,
        threaded=True)
