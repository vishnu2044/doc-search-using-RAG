# jwt_handler.py
import time
import jwt
from decouple import config

JWT_SECRET = config('SECRET_KEY')
JWT_ALGORITHM = config('ALGORITHM')

def token_response(token: str):
    return {
        'access_token': token
    }

def signJWT(userId: str):
    payload = {
        "userId": userId,
        'expiry': time.time() + 600
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token_response(token)

def decodeJWT(token: str):
    try:
        decode_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if decode_token['expiry'] >= time.time():
            return decode_token
        else:
            return None
    except jwt.ExpiredSignatureError:
        return {"info": "Token expired"}
    except jwt.InvalidTokenError:
        return {"info": "Invalid token"}
