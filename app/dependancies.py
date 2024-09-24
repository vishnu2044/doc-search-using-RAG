from fastapi import Header, HTTPException
from auth.jwt_handler import decodeJWT

def get_current_user(authorization: str = Header(...)):
    
    token = authorization.split(" ")[1] 
    decoded_token = decodeJWT(token)
    if isinstance(decoded_token, dict) and 'userId' in decoded_token:
        return decoded_token['userId']
    raise HTTPException(status_code=401, detail='Invalid or expired token')
