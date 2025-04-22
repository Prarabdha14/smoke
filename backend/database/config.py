import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:1546@localhost/smoke_detection'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
