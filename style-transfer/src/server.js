'use strict';
var express = require('express');
var request = require('request');
var cookieParser = require('cookie-parser')

var app = express();
app.use(express.static(__dirname, {'index': ['index.html']}))
   .use(cookieParser());

app.get('/', function(req, res){
  res.redirect('index.html')
});

app.set('port', (process.env.PORT || 8889));
app.listen(app.get('port'), function() {
  console.log('Node app is running on port', app.get('port'));
});
