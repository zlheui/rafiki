import os

from rafiki.utils.log import configure_logging
from rafiki.feedback.app import app

configure_logging('feedback')

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=os.getenv('FEEDBACK_PORT', 8006), 
        debug=True,
        threaded=True)
