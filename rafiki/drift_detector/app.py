from flask import Flask, request, jsonify
import os
import traceback

from rafiki.constants import UserType
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.utils.auth import generate_token, decode_token, UnauthorizedException, auth

from .drift_detector import Drift_Detector

drift_detector = Drift_Detector()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Rafiki Concept Drift Detector is up.'

@app.route('/tokens', methods=['POST'])
def generate_user_token():
    params = get_request_params()

    # Only superadmin can authenticate (other users must use Rafiki Admin)
    if not (params['email'] == SUPERADMIN_EMAIL and \
            params['password'] == SUPERADMIN_PASSWORD):
        raise UnauthorizedException()
    
    auth = {
        'user_type': UserType.SUPERADMIN
    }
    
    token = generate_token(auth)

    return jsonify({
        'user_type': auth['user_type'],
        'token': token
    })


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello':'hello'})


@app.route('/subscribe/<train_job_id>/<detector_name>', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def subscribe(auth, train_job_id, detector_name):
    with drift_detector:
        return jsonify(drift_detector.subscribe_detector(train_job_id=train_job_id, detector_name=detector_name))


@app.route('/detectors', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_detector(auth):
    params = get_request_params()
    detector_file_bytes = request.files['detector_file_bytes'].read()
    params['detector_file_bytes'] = detector_file_bytes

    with drift_detector:
        return jsonify(drift_detector.create_detector(auth['user_id'], **params))



# Handle uncaught exceptions with a server error & the error's stack trace (for development)
@app.errorhandler(Exception)
def handle_error(error):
    return traceback.format_exc(), 500

# Extract request params from Flask request
def get_request_params():
    # Get params from body as JSON
    params = request.get_json()

    # If the above fails, get params from body as form data
    if params is None:
        params = request.form.to_dict()

    # Merge in query params
    query_params = {
        k: v
        for k, v in request.args.items()
    }
    params = {**params, **query_params}

    return params