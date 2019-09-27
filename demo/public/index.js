"use strict";

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

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

function descale(points, normalizer) {
  return points.map(function (point, index, array) {
    var _point4 = _slicedToArray(point, 4),
        x = _point4[0],
        y = _point4[1],
        t = _point4[2],
        eos = _point4[3];

    var maxW = normalizer.muX * 2.25;
    var f = 1600 / maxW;
    x = x / f;
    y = y / f;
    return [x, y, t, eos];
  });
}

var Canvas =
/*#__PURE__*/
function () {
  function Canvas() {
    _classCallCheck(this, Canvas);

    this.canvas = document.getElementById('my_canvas');
    this.points = [];
  }

  _createClass(Canvas, [{
    key: "clear",
    value: function clear() {
      var ctx = this.canvas.getContext('2d');
      ctx.clearRect(0, 0, 1600, 300);
      ctx.beginPath();
      this.points = [];
    }
  }, {
    key: "addFirstPoint",
    value: function addFirstPoint(p) {
      var _p = _slicedToArray(p, 4),
          x0 = _p[0],
          y0 = _p[1],
          t0 = _p[2],
          eos0 = _p[3];

      var ctx = this.canvas.getContext('2d');
      ctx.moveTo(x0, y0);
      this.points.push(p);
    }
  }, {
    key: "addPoint",
    value: function addPoint(p, newStroke) {
      var _p2 = _slicedToArray(p, 4),
          x = _p2[0],
          y = _p2[1],
          t = _p2[2],
          eos = _p2[3];

      var ctx = this.canvas.getContext('2d');

      if (newStroke) {
        if (this.points.length > 0) {
          this.points[this.points.length - 1][3] = 1;
        }

        ctx.moveTo(x, y);
      }

      ctx.lineTo(x, y);
      ctx.stroke();
      this.points.push(p);
    }
  }, {
    key: "getPoints",
    value: function getPoints() {
      return this.points.slice(0);
    }
  }]);

  return Canvas;
}();

var canvas;

var drawPoints = function drawPoints(points, normalizer) {
  var originalPoints = points;
  var points = shiftPoints(scale(points, normalizer));

  var _points$ = _slicedToArray(points[0], 4),
      x0 = _points$[0],
      y0 = _points$[1],
      t0 = _points$[2],
      eos0 = _points$[3];

  var offsetPoints = points;
  canvas.clear();
  canvas.addFirstPoint(points[0]);

  var drawStroke = function drawStroke(index) {
    if (index >= offsetPoints.length) {
      return;
    }

    var p = offsetPoints[index];
    var prevEndOfStroke = offsetPoints[index - 1][3];
    var newStroke = prevEndOfStroke == 1;
    canvas.addPoint(p, newStroke);

    var _p3 = _slicedToArray(p, 4),
        x = _p3[0],
        y = _p3[1],
        t = _p3[2],
        eos = _p3[3];

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

var globalNormalizer;

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
    var obj = s;
    var points = obj.points;
    drawPoints(points, obj.normalizer);
    globalNormalizer = obj.normalizer;
    $('#ground_true').text(obj.transcription);
  });
};

function fetchNormalizer() {
  var params = {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  };
  fetch('http://localhost:8080/get_normalizer', params).then(function (response) {
    return response.json();
  }).then(function (obj) {
    globalNormalizer = obj.normalizer;
  });
}

function recognize(points) {
  var body = JSON.stringify({
    'line': points
  });
  var params = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: body
  };
  fetch('http://localhost:8080/recognize', params).then(function (response) {
    return response.json();
  }).then(function (s) {
    var obj = s;
    $('#predicted').text(obj.prediction);
  });
}

$(document).ready(function (e) {
  canvas = new Canvas();
  fetchNormalizer();
  var nextButton = $('#next_example_button').on('click', function (e) {
    fetchExample();
  });
  var recognizeButton = $('#recognize_button').on('click', function (e) {
    var points = descale(canvas.getPoints(), globalNormalizer);
    var first = points[0];
    var offsetPoints = points.map(function (point, index, array) {
      var _point5 = _slicedToArray(point, 4),
          x = _point5[0],
          y = _point5[1],
          t = _point5[2],
          eos = _point5[3];

      return [x - first[0], y - first[1], t - first[2], eos];
    });
    recognize(offsetPoints);
  });
  var clearButton = $('#clear_button').on('click', function (e) {
    canvas.clear();
  });
  var drawing = false;
  var first = true;

  function getPoint(e) {
    var rect = $('#my_canvas')[0].getBoundingClientRect();
    var t = Date.now() / 1000;
    return [e.clientX - rect.x, e.clientY - rect.y, t, 0];
  }

  $('#my_canvas').on('mousedown', function (e) {
    drawing = true;
    var p = getPoint(e);

    if (first) {
      var newStroke = false;
      canvas.addFirstPoint(p, false);
    } else {
      var _newStroke = true;
      canvas.addPoint(p, _newStroke);
    }

    first = false;
  });
  $('#my_canvas').on('mousemove', function (e) {
    if (drawing) {
      var p = getPoint(e);
      canvas.addPoint(p);
    }
  });
  $('#my_canvas').on('mouseup', function (e) {
    drawing = false;
  });
});