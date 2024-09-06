from fastapi import Header, HTTPException, Depends
from auth.jwt_handler import decodeJWT

def get_current_user(authorization: str = Header(...)):
    # Extract the token from the Authorization header
    token = authorization.split(" ")[1]  # Assuming the header is in the format "Bearer <token>"
    decoded_token = decodeJWT(token)
    if isinstance(decoded_token, dict) and 'userId' in decoded_token:
        return decoded_token['userId']
    raise HTTPException(status_code=401, detail='Invalid or expired token')
