from pydantic import BaseModel, EmailStr


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class SignUpRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    token: str
    user: UserOut
