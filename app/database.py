from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mindvault.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class CheckIn(Base):
    __tablename__ = "checkins"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(String, default="default", index=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    transcript      = Column(Text)
    duration_sec    = Column(Float)

    # Acoustic features
    pitch_mean      = Column(Float)
    pitch_std       = Column(Float)
    energy_mean     = Column(Float)
    speech_rate     = Column(Float)
    mfcc_mean       = Column(String)  # JSON array

    # Sentiment scores
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    acoustic_mood   = Column(Float)   # 0-1 composite
    composite_mood  = Column(Float)   # final score

    notes           = Column(Text, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()