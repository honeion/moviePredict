var express = require('express');
// var routes = require('./routes');
// var user = require('./routes/user');
var http = require('http');
var path = require('path');

var mongoose = require('mongoose');
var MemoSchema = mongoose.Schema({ username: String, memo: String });
var Memo = mongoose.model('test', MemoSchema);

var app = express();

app.set('port', process.env.PORT || 3000);
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
app.use(express.logger('dev'));
app.use(express.json());
app.use(express.urlencoded());
app.use(express.methodOverride());
app.use(app.router);
app.use(express.static(path.join(__dirname, 'public')));

app.post('/insert', function(req, res, err) {
  var memo = new Memo({ username: req.body.username, memo: req.body.memo });
  memo.save(function(err, silence) {
    if (err) {
      console.err(err);
      throw err;
    }
    res.send('success');
  });
});

const url = 'mongodb://localhost:27017';

mongoose.connect(
  url,
  function(err) {
    if (err) {
      console.log('mongoose connection error :' + err);
      throw err;
    }
    http.createServer(app).listen(app.get('port'), function() {
      console.log('Express server listening on port ' + app.get('port'));
    });
  }
);

// app.get('/users/:username', function(req, res, err) {
//   var memos = new Memo();
//   Memo.findOne({ username: req.params.username }, function(err, memo) {
//     if (err) {
//       console.err(err);
//       throw err;
//     }
//     console.log(memo);
//     res.send(200, memo);
//   });
// });
