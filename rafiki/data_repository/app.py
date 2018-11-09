from flask import Flask, request, jsonify
import os
import traceback

from rafiki.constants import UserType, ServiceType
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.utils.auth import generate_token, decode_token, UnauthorizedException, auth

from .data_repository import DataRepository

data_repository = DataRepository()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Rafiki Data Repository Server is up.'

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

@app.route('/print_folder_structure/<train_job_id>', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def print_folder_structure(auth, train_job_id):
    with data_repository:
        return jsonify(data_repository.print_folder_structure(train_job_id=train_job_id))

@app.route('/remove_folder/<train_job_id>', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def remove_train_job_folder(auth, train_job_id):
    with data_repository:
        return jsonify(data_repository.remove_train_job_folder(train_job_id=train_job_id))

@app.route('/remove_all_folders', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def remove_all_folders(auth):
    with data_repository:
        return jsonify(data_repository.remove_all_folders())

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello':'hello'})

@app.route('/data_repository/query', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_data_repository_query_service(auth):
    with data_repository:
        return jsonify(data_repository.create_data_repository_service(ServiceType.REPOSITORY_QUERY))

@app.route('/data_repository/stop/query', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def stop_data_repository_query_service(auth):
    with data_repository:
        return jsonify(data_repository.stop_data_repository_service(ServiceType.REPOSITORY_QUERY))

@app.route('/data_repository/feedback', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_data_repository_feedback_service(auth):
    with data_repository:
        return jsonify(data_repository.create_data_repository_service(ServiceType.REPOSITORY_FEEDBACK))

@app.route('/data_repository/stop/feedback', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def stop_data_repository_feedback_service(auth):
    with data_repository:
        return jsonify(data_repository.stop_data_repository_service(ServiceType.REPOSITORY_FEEDBACK))

@app.route('/create_new_dataset/<train_job_id>', methods=['GET'])
def create_new_dataset(auth, train_job_id):
    params = get_request_params()
    with data_repository:
        return jsonify(data_repository.create_new_dataset(train_job_id, **params))

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
