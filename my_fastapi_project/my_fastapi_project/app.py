from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import joblib

class PassengerData(BaseModel):
    PassengerId: str
    HomePlanet: Optional[str]
    CryoSleep: Optional[bool]
    Cabin: Optional[str]
    Destination: Optional[str]
    Age: Optional[float]
    VIP: Optional[bool]
    RoomService: Optional[float]
    FoodCourt: Optional[float]
    ShoppingMall: Optional[float]
    Spa: Optional[float]
    VRDeck: Optional[float]
    Name: Optional[str]

app = FastAPI()

model = joblib.load('trained_model.pkl') # Load the trained model

def redata(train_data, test):
    columns_to_fill = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    train_data[columns_to_fill] = train_data[columns_to_fill].fillna(0)

    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

    train_data['Deck'] = train_data['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 0)
    train_data['Num'] = train_data['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else 0)
    train_data['Side'] = train_data['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 0)

    train_data['TotalExpenses'] = (train_data['RoomService'] + train_data['FoodCourt'] + train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck'])
    
    train_data = train_data.drop(['Cabin', 'Name', 'PassengerId'], axis=1)

    train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], drop_first=True)
    train_data = train_data.astype(int)
    
    if test == 0:
        DataTrain = train_data.drop('Transported', axis=1)
        ResDataTrain = train_data['Transported']
        return [DataTrain, ResDataTrain]
    return train_data

# Read and prepare training data
train_data = pd.read_csv('train.csv')
DataTrain, _ = redata(train_data, test=0)

@app.post("/predict")
async def predict(passengers: List[PassengerData]):
    passengers_data = [passenger.dict() for passenger in passengers]
    test_data = pd.DataFrame(passengers_data)
    passenger_ids = test_data['PassengerId']
    test_data_prepared = redata(test_data, test=1)

    # Align columns with training data
    missing_cols = set(DataTrain.columns) - set(test_data_prepared.columns)
    for col in missing_cols:
        test_data_prepared[col] = 0
    test_data_prepared = test_data_prepared[DataTrain.columns]

    predictions = model.predict(test_data_prepared) # Make predictions

    output = pd.DataFrame({ 
        'PassengerId': passenger_ids, 
        'Transported': predictions})
    output['Transported'] = output['Transported'].apply(lambda x: 'True' if x == 1 else 'False')

    return JSONResponse(content=output.to_dict(orient='records'))

@app.post("/get_model")
async def get_model():
    return FileResponse('trained_model.pkl', 
                        media_type='application/octet-stream', 
                        filename='trained_model.pkl')

def prepare_data(file: UploadFile):
    test_data = pd.read_csv(file.file)
    train_data = pd.read_csv('train.csv')
    DataTrain = redata(train_data, 0)[0]

    # Преобразуем данные через функцию в другом файле
    test_data_prepared = redata(test_data, 1)
    
    # колонки совпадают
    missing_cols = set(DataTrain.columns) - set(test_data_prepared.columns)
    for col in missing_cols: test_data_prepared[col] = 0
    test_data_prepared = test_data_prepared[DataTrain.columns]
    
    return [test_data_prepared, test_data]

@app.post("/predict_file")
async def predict(file: UploadFile = File(...)):
    [data, test_data] = prepare_data(file)
    # passenger_ids = test_data['PassengerId']

    predictions = model.predict(data)
    # Создание DataFrame для результата
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': predictions})
    submission['Transported'] = submission['Transported'].apply(lambda x: 'True' if x == 1 else 'False')

    submission_file_path = "submission.csv"
    submission.to_csv(submission_file_path, index=False)
    return FileResponse(path=submission_file_path, media_type='text/csv', filename="submission.csv")