"use strict";

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance"); }

function _iterableToArrayLimit(arr, i) { if (!(Symbol.iterator in Object(arr) || Object.prototype.toString.call(arr) === "[object Arguments]")) { return; } var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

function shiftPoints(points) {
  var minX = 10000;
  var minY = 10000;
  points.forEach(function (point, index, array) {
    var _point = _slicedToArray(point, 4),
        x = _point[0],
        y = _point[1],
        t = _point[2],
        eos = _point[3];

    if (x < minX) {
      minX = x;
    }

    if (y < minY) {
      minY = y;
    }
  });
  return points.map(function (point, index, array) {
    var _point2 = _slicedToArray(point, 4),
        x = _point2[0],
        y = _point2[1],
        t = _point2[2],
        eos = _point2[3];

    return [x - minX, y - minY, t, eos];
  });
}

function scale(points, normalizer) {
  return points.map(function (point, index, array) {
    var _point3 = _slicedToArray(point, 4),
        x = _point3[0],
        y = _point3[1],
        t = _point3[2],
        eos = _point3[3];

    var maxW = normalizer.muX * 2.25;
    var f = 1600 / maxW;
    x = x * f;
    y = y * f;
    return [x, y, t, eos];
  });
}

var drawPoints = function drawPoints(points, normalizer) {
  var canvas = document.getElementById('my_canvas');
  var ctx = canvas.getContext('2d');
  var points = shiftPoints(scale(points, normalizer));

  var _points$ = _slicedToArray(points[0], 4),
      x0 = _points$[0],
      y0 = _points$[1],
      t0 = _points$[2],
      eos0 = _points$[3];

  var offsetPoints = points;
  ctx.moveTo(x0, y0);

  var drawStroke = function drawStroke(index) {
    if (index >= offsetPoints.length) {
      return;
    }

    var _offsetPoints$index = _slicedToArray(offsetPoints[index], 4),
        x = _offsetPoints$index[0],
        y = _offsetPoints$index[1],
        t = _offsetPoints$index[2],
        eos = _offsetPoints$index[3];

    var prevEndOfStroke = offsetPoints[index - 1][3];

    if (prevEndOfStroke == 1) {
      ctx.moveTo(x, y);
    }

    ctx.lineTo(x, y);
    ctx.stroke();
    var dt = offsetPoints[index] - offsetPoints[index - 1];
    var delayInSeconds = dt * 1000;
    setTimeout(function () {
      return drawStroke(index + 1);
    }, delayInSeconds);
  };

  var _offsetPoints$ = _slicedToArray(offsetPoints[0], 4),
      x0 = _offsetPoints$[0],
      y0 = _offsetPoints$[1],
      t0 = _offsetPoints$[2],
      eos0 = _offsetPoints$[3];

  var _offsetPoints$2 = _slicedToArray(offsetPoints[1], 4),
      x1 = _offsetPoints$2[0],
      y1 = _offsetPoints$2[1],
      t1 = _offsetPoints$2[2],
      eos1 = _offsetPoints$2[3];

  setTimeout(function () {
    return drawStroke(1);
  }, t1 - t0);
};

var fetchExample = function fetchExample() {
  var params = {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  };
  fetch('http://localhost:8080/get_example', params).then(function (response) {
    return response.json();
  }).then(function (s) {
    console.log(s);
    var obj = s;
    var points = obj.points;
    console.log(obj.transcription);
    drawPoints(points, obj.normalizer);
  });
};

setTimeout(fetchExample, 5000);