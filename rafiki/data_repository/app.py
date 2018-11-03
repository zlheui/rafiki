from flask import Flask, request, jsonify
import os
import traceback

from rafiki.constants import UserType
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


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello':'hello'})


@app.route('/<train_job_id>/', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_new_dataset(auth, train_job_id):
    params = get_request_params()
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
