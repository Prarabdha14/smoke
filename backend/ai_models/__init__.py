from sqlalchemy import Column, Integer, String
from database.db import Base  # or wherever your SQLAlchemy Base is defined

class ImageRecord(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    status = Column(String)
    timestamp = Column(String)
