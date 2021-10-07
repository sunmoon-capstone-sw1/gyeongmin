from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
import pyttsx3


# define new database
db = SQLAlchemy()
DB_NAME = "img.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'abc'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    # import .models as models
    from .models import User, Note
    create_database(app)
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    # telling flask how we load a user
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id)) # look up at primarykey


    return app

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        # print('Created Database!')

def text_to_speech(text):
     engine = pyttsx3.init()
     engine.setProperty('rate', 125)
     engine.setProperty('volume', 1.0)

     engine.say(text)
     engine.runAndWait()

