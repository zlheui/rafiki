import os

from rafiki.utils.log import configure_logging
from rafiki.data_repository.app import app

configure_logging('data_repository')

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=os.getenv('DATA_REPOSITORY_PORT', 8007), 
        debug=True,
        threaded=True)
