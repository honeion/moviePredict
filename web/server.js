var express = require('express');
var app = express();
var bodyParser = require('body-parser');
var urlencode = require('urlencode');
let { PythonShell } = require('python-shell');
// var router = require('./router/main')(app);

출처: http: app.set('views', __dirname + '/views');
app.set('view engine', 'jade');
app.locals.pretty = true;
app.engine('html', require('ejs').renderFile);

app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: false }));

///////////////////////// https://www.npmjs.com/package/mongodb
const MongoClient = require('mongodb').MongoClient;
const Server = require('mongodb').Server;
const assert = require('assert');

// Connection URL
const url = 'mongodb://localhost:27017';

// Database Name
const dbName = 'moviePredict';

// DB에서 정보 가져오기
var movieNmArray = []; //모든 영화의 이름을 저장할 배열
var allDocument; //DB에 들어있는 모든 정보를 저장할 변수
var allDocument_test; //DB에 들어있는 test Set 정보를 저장할 변수
MongoClient.connect(
  url,
  function(err, client) {
    const db = client.db(dbName);
    var collection = db.collection('moviecollection2');
    collection.find({}).toArray(function(err, docs) {
      allDocument = docs;
      for (var i = 0; i < docs.length; i++) {
        movieNmArray[i] = docs[i].movieNm;
      }
    });
    var collection2 = db.collection('moviecollection_test');
    collection2.find({}).toArray(function(err, docs) {
      allDocument_test = docs.slice(0, docs.length - 1); //DB 마지막에 null값이 들어있어서 -1을 해줌
    });
  }
);

//서버 실행 & 첫페이지에 보여줄 영화 정보 가져와 뿌리기
var queryLimit = 12; //home화면에서 보여줄 영화 개수
//파라미터로 포트번호와 IP번호(String값으로), IP번호 적지 않으면 localhost
var server = app.listen(3000, function() {
  console.log('Express server has started on port 3000');
  app.get('/home', function(req, res) {
    res.render('home', {
      //home.jade로 넘길 변수들 선언
      database: allDocument.slice(0, queryLimit), //한 페이지에 12개만 보여주므로 슬라이스해서 넘김
      queryLimit,
      dblength: movieNmArray.length,
      currentPageNumber: 1, //화면에 보여줄 페이지 번호(첫페이지이므로 1)
    });
  });
});

//포스터, 영화이름, Read More 버튼 클릭했을 때 해당 영화 페이지로 이동
app.post('/gotoMovieInfo', function(req, res) {
  var document = req.body.clickedMovie.split(','); //post방식으로 받은 영화정보가 ,로 구분된 String형식이므로 스플릿함.
  console.log(document[4]);
  app.get('/' + document[0].toString(), function(req, res) {
    res.render('movie_info', {
      //movie_info.jade로 넘길 변수들(줄거리, 포스터링크, 비디오링크, 영화이름, 감독, 배우들, 장르, 상영시간, 등급, 개봉일, 예측관객수, 실제관객수, 예측률, 예측그래프, 개봉전예측)
      story: document[1].split('<br/>?'),
      poster: document[2],
      videoLink: document[3],
      movieNm: document[4],
      director: document[5],
      actor1: document[6],
      actor2: document[7],
      actor3: document[8],
      nation: document[9],
      genre: document[10],
      showTm: document[11],
      watchGrade: document[12],
      openDt: document[13],
      prediction: document[14],
      realValue: document[15],
      accuracy: document[16],
      predictionGraph: document[17],
      D0_prediction: document[18],
    });
  });
  res.redirect('/' + document[0].toString());
});

//영화 검색시 해당 영화 페이지로 이동
app.post('/search', function(req, res) {
  var inputText = req.body.searchContent; //검색한 내용
  console.log(inputText);
  var tempIndex = [];
  var count = 0;
  //검색한 내용이 영화이름에 포함되면 인덱스 저장
  for (var i = 0; i < movieNmArray.length - 1; i++) {
    var boolFlag = movieNmArray[i].indexOf(req.body.searchContent);
    if (boolFlag != -1) {
      tempIndex[count] = i;
      count++;
    }
  }
  var database = [];
  count = 0;
  //DB정보에서 인덱스로 접근하여 database 변수에 저장
  for (var i = 0; i < tempIndex.length; i++) {
    database[count] = allDocument[tempIndex[i]];
    count++;
  }

  app.get('/search_' + encodeURI(encodeURIComponent(inputText)), function(
    req,
    res
  ) {
    res.render('search_result', {
      //search_result.jade로 검색 결과의 영화정보와 검색어를 넘김
      database,
      inputText,
    });
  });
  res.redirect('/search_' + encodeURI(encodeURIComponent(inputText)));
});

//페이지 번호 클릭시 해당 페이지 번호로 이동(그에 맞는 영화 정보 뿌리기)
app.post('/changePage', function(req, res) {
  var currentPageNumber = req.body.clickedNumber; //클릭한 페이지 번호
  app.get('/home' + currentPageNumber, function(req, res) {
    res.render('home', {
      //그 번호에 맞게 화면에 보여줄 영화 정보 슬라이스
      database: allDocument.slice(
        queryLimit * (currentPageNumber - 1),
        queryLimit * currentPageNumber
      ),
      queryLimit,
      dblength: movieNmArray.length,
      currentPageNumber: parseInt(currentPageNumber),
    });
  });
  res.redirect('/home' + currentPageNumber);
});

module.exports = app;

//새로운 test set 페이지
app.get('/newest', function(req, res) {
  res.render('test_result', {
    database: allDocument_test,
  });
});

//변인별 영향력 그래프를 보여줄 페이지
app.get('/about', function(req, res) {
  res.render('about.html');
});

//예측 관객수를 구하기 위해 사용자가 변인들을 입력하는 페이지
app.get('/prediction', function(req, res) {
  res.render('prediction');
});

//사용자가 영화 변인들을 입력하고 예측버튼을 누르면 파이썬 코드 실행하여 예측값 보여줄 페이지로 이동
app.post('/predictionStart', function(req, res) {
  var parameter = [
    req.body.movieNm,
    req.body.director,
    req.body.actor1,
    req.body.actor2,
    req.body.actor3,
    req.body.actor4,
    req.body.actor5,
    req.body.actor6,
    req.body.year,
    req.body.month,
    req.body.genre,
    req.body.showTm,
    req.body.grade,
    req.body.nation,
    req.body.company,
    req.body.starScore,
    req.body.userCount,
    req.body.screen,
    req.body.audience,
    req.body.show,
  ];
  var encodingParameter = [];
  var dataString = '';
  for (var i = 0; i < 20; i++) {
    // 한글 인코딩 후 파이썬으로 넘김, 파이썬에서는 받은 값 디코딩하면 한글로 바르게 출력됨
    encodingParameter[i] = urlencode.encode(parameter[i]);
    dataString += parameter[i]; //예측값 보여줄 페이지주소 만들기 위한 값
  }

  let options = {
    mode: 'text',
    // pythonPath: 'path/to/python',
    pythonOptions: ['-u'], // get print results in real-time
    // scriptPath: 'path/to/my/scripts',
    args: encodingParameter,
  };

  //파이썬 파일 실행
  PythonShell.run('parameter_prediction.py', options, function(err, results) {
    if (err) throw err;

    console.log(urlencode.decode(results[0]), dataString);

    // var predictionResult = urlencode.decode(results[0]);
    app.get('/test' + encodeURI(encodeURIComponent(dataString)), function(
      req,
      res
    ) {
      res.render('test', {
        //test.jade로 넘길 변수들 선언
        predictionResult: results[0],
        movieNm: parameter[0],
        director: parameter[1],
        actor1: parameter[2],
        actor2: parameter[3],
        actor3: parameter[4],
        actor4: parameter[5],
        actor5: parameter[6],
        actor6: parameter[7],
        year: parameter[8],
        month: parameter[9],
        genre: parameter[10],
        showTm: parameter[11],
        grade: parameter[12],
        nation: parameter[13],
        company: parameter[14],
        starScore: parameter[15],
        userCount: parameter[16],
        scree: parameter[17],
        audience: parameter[18],
        show: parameter[19],
      });
    });
    res.redirect('/test' + encodeURI(encodeURIComponent(dataString)));
  });

  // res.redirect('/prediction');
});
