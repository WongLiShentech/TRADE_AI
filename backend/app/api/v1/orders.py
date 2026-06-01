from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.order import OrderCreate, OrderRead

router = APIRouter()


@router.get("", response_model=list[OrderRead])
def list_orders(db: Session = Depends(get_db)):
    raise NotImplementedError


@router.post("", response_model=OrderRead, status_code=201)
def create_order(order: OrderCreate, db: Session = Depends(get_db)):
    raise NotImplementedError
