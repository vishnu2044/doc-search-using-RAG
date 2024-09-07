from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from model import User, UserSchema, UserLoginSchema, SessionLocal
from auth.jwt_handler import signJWT
from fastapi import FastAPI, File, UploadFile
from doc_chat.initial_setup import process_document
from dependancies import get_current_user

from fastapi import HTTPException, Query
import os
from doc_chat.initial_setup import load_and_query_chroma_db
from starlette.middleware.cors import CORSMiddleware


app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

#Signup for new user
@app.post('/user/signup/', tags=['user'])
def user_signup(user: UserSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = User(**user.dict())
    db.add(new_user)
    db.commit()
    return signJWT(user.email)

#Login user for current user
@app.post('/user/login/', tags=['user'])
def user_login(user: UserLoginSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email, User.password == user.password).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    print("user email ::::::",user.email)
    return signJWT(user.email)


# Upload documents
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), user_email: str = Depends(get_current_user)):
    temp_file_path = f"temp_{file.filename}"
    print("file_name :::", file.filename)
    file_name = file.filename
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    print("Document is added to processing ::::::::::::::::::::::::::::::::::::::")
    process_document(temp_file_path, user_email, file_name)

    os.remove(temp_file_path)

    return {"message": "document addedd successfully !"}


# enquiry to get the aspect query
@app.put('/doc_enqury/')
async def doc_q_a(query_text: str = Query(...), user_email: str = Depends(get_current_user)):
    print("The function is started to working :::::::::::::::::::::::::::::::::::::")
    if not query_text or not user_email:
        raise HTTPException(status_code=400, detail="Query text and user_email are required.")
    response = load_and_query_chroma_db(query_text)

    return {"response": response}



