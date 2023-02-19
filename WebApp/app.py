from datetime import datetime
from flask import Flask, render_template, flash, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, validators, PasswordField
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from networks import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
inv_normalize = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2))
])


app = Flask(__name__)
app.config['SECRET_KEY'] = 'abcdefgh'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

app.config['UPLOAD_FOLDER'] = "static"

# Flask_Login Stuff
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Registration(FlaskForm):
    username = StringField('Username', validators=[validators.InputRequired(), validators.Length(min=5, max=20)])
    firstname = StringField('Firstname', validators=[validators.InputRequired(), validators.Length(min=3, max=20)])
    lastname = StringField('Lastname', validators=[validators.InputRequired(), validators.Length(min=1, max=20)])
    address = StringField('Address', validators=[validators.InputRequired(), validators.Length(min=5, max=100)])
    email = StringField('Email', validators=[validators.InputRequired(), validators.Length(min=5, max=30), validators.Email()])
    password = PasswordField('Password', validators=[validators.InputRequired(), validators.Length(min=6, max=30), validators.EqualTo('password2', message='Passwords did not match !!!')])
    password2 = PasswordField('Confirm', validators=[validators.InputRequired(), validators.Length(min=6, max=30)])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[validators.InputRequired(), validators.Length(min=5, max=20)])
    password = PasswordField('Password', validators=[validators.InputRequired(), validators.Length(min=6, max=30)])
    submit = SubmitField('Sign In')
    
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    firstname = db.Column(db.String(20), unique=False, nullable=False)
    lastname = db.Column(db.String(20), unique=False, nullable=False)
    address = db.Column(db.String(50), unique=False, nullable=False)
    email = db.Column(db.String(30), unique=True, nullable=False)
    hashed_password = db.Column(db.String(15), unique=False, nullable=False)
    date_added = db.Column(db.DateTime, default = datetime.utcnow)

    def __repr__(self):
        return f"{self.username}, {self.firstname}, {self.lastname}, {self.address}, {self.email}"

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')


    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


@app.route("/", methods=['GET', 'POST'])
def login():
    if os.path.exists("static/generated.jpg"):
        print("delete login")
        os.remove("static/to-generate.jpg")
        os.remove("static/generated.jpg")

    form = LoginForm()
    username = None
    if form.validate_on_submit():
        user = User.query.filter_by(username = form.username.data).first()
        if user:
            # Check Hash
            if check_password_hash(user.hashed_password, form.password.data):
                login_user(user)
                flash("Login Successfull !!!")
                return redirect(url_for(f'dashboard', username = form.username.data))                    
            else:
                flash("Wrong Password !!!")
        else:
            flash("User does not exist !!!")

        form.password.data = ''
        form.username.data = ''
    return render_template("login.html", username = username, form=form)

# Create Logout
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    if os.path.exists("static/generated.jpg"):
        print("delete logout")
        os.remove("static/to-generate.jpg")
        os.remove("static/generated.jpg")

    logout_user()
    flash("You have been Logged Out !!!")
    return redirect(url_for(f'login'))

# Create profile
@app.route('/profile/<username>', methods=['GET', 'POST'])
@login_required
def profile(username):
    if os.path.exists("static/generated.jpg"):
        os.remove("static/to-generate.jpg")
        os.remove("static/generated.jpg")
    user = User.query.filter_by(username=username).first()
    return render_template("profile.html", username = username, user = user)

# About Us
@app.route('/about', methods=['GET', 'POST'])
def about():
    if os.path.exists("static/generated.jpg"):
        os.remove("static/to-generate.jpg")
        os.remove("static/generated.jpg")
    return render_template("about.html")

# Dashboard
@app.route("/dashboard/<username>", methods=['GET', 'POST'])
@login_required
def dashboard(username):
    if request.method=='POST':
        if os.path.exists("static/generated.jpg"):
            os.remove("static/to-generate.jpg")
            os.remove("static/generated.jpg")

        contentFile = request.files['contentImg']
        if contentFile :
            contentFile.save(os.path.join(app.config['UPLOAD_FOLDER'], "to-generate.jpg"))
            flash("Uploaded Successfully !!!")
            return render_template('dashboard.html', username = username)
        else:
            flash("Image not selected !!!")
            return render_template('dashboard.html', username = username)

    return render_template("dashboard.html", username = username)

@app.route("/imageGenerator/<username>", methods=['GET','POST'])
def imageGenerator(username):
    inputData = dict(request.form)
    if inputData['style'] == 'select':
        flash("Style not selected !!!")
        return redirect(url_for(f'dashboard', username = username))
        
    if not os.path.exists("static/to-generate.jpg"):
        flash("Image not selected !!!")
        return redirect(url_for(f'dashboard', username = username))

    model = Generator()
    if inputData['style'] == 'monet':
        model.load_state_dict(torch.load("static/Trained Models/photo2monet-max-epoch-200.pt", map_location = torch.device('cpu')))
    if inputData['style'] == 'cezanne':
        model.load_state_dict(torch.load("static/Trained Models/photo2cezanne-max-epoch-200.pt", map_location = torch.device('cpu')))
    if inputData['style'] == 'vangogh':
        model.load_state_dict(torch.load("static/Trained Models/photo2vangogh-max-epoch-200.pt", map_location = torch.device('cpu')))
    if inputData['style'] == 'ukiyoe':
        model.load_state_dict(torch.load("static/Trained Models/photo2ukiyoe-max-epoch-200.pt", map_location = torch.device('cpu')))
    
    model.to(device)
    model.eval()

    files = "static/to-generate.jpg"
    img = Image.open(os.path.join(files)).convert("RGB")
    if img.size[0] > 1280 or img.size[1] > 768: 
        img = img.resize((1280, 768), Image.LANCZOS)

    img = transform(img)
    img = img.unsqueeze(0)

    out = model(img.to(device))
    img = img.detach().cpu()

    out = inv_normalize(out)
    temp = out.squeeze(0)
    temp = temp.detach()
    temp = temp.cpu().permute(1, 2, 0).numpy()
    plt.imsave("static/generated.jpg", temp)

    print(inputData)
    if request.method=='POST':
        return redirect(url_for(f'dashboard', username = username))
    if request.method=='POST' and not os.path.exists(os.path.join("static", "to-generate.jpg")):
        return redirect(url_for(f'dashboard', username = username))


@app.route("/signup", methods=['GET', 'POST'])
def signup():
    username = None
    form = Registration()    
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        email = User.query.filter_by(email=form.email.data).first()

        if not user and not email:
            username = form.username.data
            add_user = User(username = form.username.data, firstname = form.firstname.data, lastname = form.lastname.data,
            address = form.address.data, email = form.email.data, hashed_password = generate_password_hash(form.password.data, "sha256"))
            db.session.add(add_user)
            db.session.commit()
            flash("User Registered Succesfully")

        else:
            flash("Username or Email Address already present !!!")
        
        form.username.data = ''
        form.firstname.data = ''
        form.lastname.data = ''
        form.email.data = ''
        form.address.data = ''
        form.password.data = ''
        form.password2.data = ''
    
    return render_template("signup.html", username = username, form = form)


# main driver function
if __name__ == '__main__':
	app.run(debug=True)
