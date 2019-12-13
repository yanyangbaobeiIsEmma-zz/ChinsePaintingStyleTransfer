from flask import Flask, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from utils import *
from model import  *
from transfer import *
from datetime import datetime
import uuid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

UPLOAD_FOLDER = 'images_dir'
PROCESSED_FOLDER = 'processed_dir'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER



# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


@app.route('/images/<filename>')
def uploaded_image_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)

@app.route('/processed/<filename>')
def uploaded_processed_file(filename):
	return send_from_directory(app.config['PROCESSED_FOLDER'],
							   filename)

@app.route('/uploads/<filename>/<outputfilename>')
def uploaded_file(filename, outputfilename):
	# return send_from_directory(app.config['PROCESSED_FOLDER'],
	# 						   filename)
	return '''
	<!doctype html>
	<html>
		<head>
			<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css" integrity="sha384-rwoIResjU2yc3z8GV/NPeZWAv56rSmLldC3R/AZzGRnGxQQKnKkoFVhFQhNUwEyJ" crossorigin="anonymous">
			<script src="https://code.jquery.com/jquery-3.1.1.slim.min.js" integrity="sha384-A7FZj7v+d/sdmMqp/nOQwliLvUsJfDHW+k9Omg/a/EheAdgtzNs3hpfag6Ed950n" crossorigin="anonymous"></script>
			<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js" integrity="sha384-DztdAPBWPRXSA/3eYEEUWrWCy7G5KFbe8fFjk5JAIxUYHKkDx6Qin1DkWx51bBrb" crossorigin="anonymous"></script>
			<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
		</head>
		<body class="text-center">
			<div class='container'>
				<div class="row">
					<div class="col-12">
						<div style='height: 50px;'></div>
						<title>Chinese Style Transfer</title>
						<h1>Chinese Style Transfer</h1>
						<div style="max-width: 200px;">
							<a href="http://20.185.103.117:5000/" class="btn btn-primary" style="width: 100%;">
								Back
							</a>
						</div>
						
					</div>
					<div class="col-4">

					</div>
					<div class="col-4">
						<img src="''' + '/images/'+ filename + '''"  class="img-thumbnail">
					</div>
					<div class="col-4">
						<img src="''' + '/processed/' + outputfilename + '''"  class="img-thumbnail">
					</div>
				</div>
			</div>
		</body>
	<html>
	'''

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			model_tyle_param = request.form.get('type')
			model_type = 'cnn-gongbi' if model_tyle_param == "1" else 'cnn-shuimo'
			output_path = transfer(os.path.join(app.config['UPLOAD_FOLDER'], filename), model_type)
			return redirect(url_for('uploaded_file', filename=filename,
									outputfilename=output_path ))
	return '''
	<!doctype html>
	<html>
		<head>
			<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css" integrity="sha384-rwoIResjU2yc3z8GV/NPeZWAv56rSmLldC3R/AZzGRnGxQQKnKkoFVhFQhNUwEyJ" crossorigin="anonymous">
			<script src="https://code.jquery.com/jquery-3.1.1.slim.min.js" integrity="sha384-A7FZj7v+d/sdmMqp/nOQwliLvUsJfDHW+k9Omg/a/EheAdgtzNs3hpfag6Ed950n" crossorigin="anonymous"></script>
			<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js" integrity="sha384-DztdAPBWPRXSA/3eYEEUWrWCy7G5KFbe8fFjk5JAIxUYHKkDx6Qin1DkWx51bBrb" crossorigin="anonymous"></script>
			<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
		</head>
		<body class="">
			<script>
				function loading() {
					document.getElementById('loading').style.display="block"; 
				}
			</script>
			<div class='container'>
				<div class="row">
					<div class="col-12 text-center">
						<div style='height: 50px;'></div>
						<title>Chinese Style Transfer</title>
						<h1>Chinese Style Transfer</h1>
						<form method=post enctype=multipart/form-data style="max-width: 200px;" id="form">
							<div class="form-group">
							  	<input class="form-control-file" type=file name=file>
							</div>
							<div class="form-group">
								<select class="form-control" id="model-type" name=type form="form">
							      <option value="1">Gongbi</option>
							      <option value="2">Shuimo</option>

							    </select>
						    </div>
		    				<div class="form-group">
							  	<input class="form-control" type=submit value=Upload onclick="loading();">
							</div>
				
						</form>
					</div>
					<div class="col-4" id="loading" style="display: none; margin-top: 20px;">
						<img src="''' + '/images/'+ 'loading.gif' + '''" style="width: 120px;"">
					</div>
					<div class="col-4">

					</div>
					<div class="col-4">

					</div>
				</div>

			</div>
		</body>
	<html>
	'''


def transfer(image_path, model_type):
	content_img = image_loader(image_path)
	content = image_path.split('/')[-1].split('.')[0]
	# model_type = 'cnn-neural', 'cnn-gongbi', 'cnn-shanshui'
	optimizer = 'SGD'
	num_steps = 10
	style_weight = 1000000
	content_weight = 1
	cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
	cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device) 
	file_path =  str(uuid.uuid1()) + ".jpg"
	output_path = os.path.join(app.config['PROCESSED_FOLDER'], file_path)
	if "gongbi" in model_type:
		style_img = image_loader('gong_bi_hua_871.jpg') 
	else:
		style_img = image_loader('shan_shui_hua_0.jpg')   

	content_layers = ['conv_4']
	style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
	optimizer = "LBFGS"
	transferModel = models.vgg19(pretrained=True).features.to(device).eval()
	if model_type == "cnn-neural": # cnn neural transfer
		input_img = content_img
	else: # naive combination
		tmp_output = "intermediate_output.png"
		if "gongbi" in model_type:
			input_img, _ = gan_generator_eval("gongbi", image_path, tmp_output)
		else:
			input_img, _ = gan_generator_eval("shuimo", image_path, tmp_output)
	output = run_style_transfer(transferModel, cnn_normalization_mean, cnn_normalization_std,
									content_img, style_img, input_img, content_layers, style_layers, 'loss.txt', 
									model_type=model_type, optimizer='LBFGS',
									num_steps=num_steps, style_weight=style_weight,
									content_weight=content_weight)

	save_image(output, output_path)
	return file_path


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)