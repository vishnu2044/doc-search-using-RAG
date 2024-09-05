from fastapi import FastAPI, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from model import User, UserSchema, UserLoginSchema, SessionLocal
from auth.jwt_handler import signJWT, decodeJWT
from fastapi import FastAPI, File, UploadFile
from chat.basicSetup import process_document

import os

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post('/user/signup', tags=['user'])
def user_signup(user: UserSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = User(**user.dict())
    db.add(new_user)
    db.commit()
    return signJWT(user.email)

@app.post('/user/login', tags=['user'])
def user_login(user: UserLoginSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email, User.password == user.password).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return signJWT(user.email)




app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Process the document using the function from docReader
    process_document(temp_file_path)

    # Optionally, remove the temporary file after processing
    os.remove(temp_file_path)

    return {"document": "document"}
