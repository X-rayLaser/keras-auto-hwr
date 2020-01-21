function shiftPoints(points) {
    let minX = 10000;
    let minY = 10000;

    points.forEach((point, index, array) => {
        let [x, y, t, eos] = point;
        if (x < minX) {
            minX = x;
        }

        if (y < minY) {
            minY = y
        }
    });

    return points.map((point, index, array) => {
        let [x, y, t, eos] = point;

        return [x - minX, y - minY, t, eos];
    });
}

function scale(points, normalizer) {
    return points.map((point, index, array) => {
        var [x, y, t, eos] = point;

        let maxW = normalizer.muX * 2.25;

        var f = 1600 / maxW;
        x = x * f;
        y = y * f;
        return [x, y, t, eos];
    });
}

function descale(points, normalizer) {
    return points.map((point, index, array) => {
        var [x, y, t, eos] = point;

        let maxW = normalizer.muX * 2.25;

        var f = 1600 / maxW;
        x = x / f;
        y = y / f;
        return [x, y, t, eos];
    });
}


class Canvas {
    constructor() {
        this.canvas = document.getElementById('my_canvas');
        this.points = [];
    }

    clear() {
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, 1600, 300);
        ctx.beginPath();
        this.points = [];
    }

    addFirstPoint(p) {
        let [x0, y0, t0, eos0] = p;
        const ctx = this.canvas.getContext('2d');
        ctx.moveTo(x0, y0);
        this.points.push(p)
    }

    addPoint(p, newStroke) {
        let [x, y, t, eos] = p;
        const ctx = this.canvas.getContext('2d');

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

    getPoints() {
        return this.points.slice(0);
    }
}


let canvas;


const drawPoints = (points, normalizer) => {
    let originalPoints = points;
    var points = shiftPoints(scale(points, normalizer));

    var [x0, y0, t0, eos0] = points[0];

    let offsetPoints = points;

    canvas.clear();
    canvas.addFirstPoint(points[0]);

    const drawStroke = index => {
        if (index >= offsetPoints.length) {
            enableButtons();
            return;
        }

        let p = offsetPoints[index];

        let prevEndOfStroke = offsetPoints[index - 1][3];
        let newStroke = (prevEndOfStroke == 1);
        canvas.addPoint(p, newStroke);

        let [x, y, t, eos] = p;
        let dt = offsetPoints[index] - offsetPoints[index - 1];
        let delayInSeconds = dt * 1000;
        setTimeout(() => drawStroke(index + 1), delayInSeconds);
    }

    var [x0, y0, t0, eos0] = offsetPoints[0]
    const [x1, y1, t1, eos1] = offsetPoints[1]

    setTimeout(() => drawStroke(1), t1 - t0);
}


let globalNormalizer;


const fetchExample = () => {
    var params = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    };
    fetch('http://localhost:8080/get_example', params).then(function (response) {
        return response.json();
    }).then(function (s) {
        let obj = s;
        let points = obj.points;
        drawPoints(points, obj.normalizer);
        globalNormalizer = obj.normalizer;
        $('#ground_true').text(obj.transcription);
    })
}

function fetchNormalizer () {
    var params = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    };
    fetch('http://localhost:8080/get_normalizer', params).then(response => {
        return response.json();
    }).then(obj => {
        globalNormalizer = obj.normalizer;
    })
}

function recognize(points) {
    let body = JSON.stringify({'line': points});
    var params = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: body
    };

    fetch('http://localhost:8080/recognize', params).then(response => {
        return response.json();
    }).then(s => {
        let obj = s;
        $('#predicted').text(obj.prediction);
        enableButtons();
    })
}

function enableButtons() {
    $('#recognize_button').attr("disabled", false);
    $('#next_example_button').attr("disabled", false);
    $('#clear_button').attr("disabled", false);
}

function disableButtons() {
    $('#recognize_button').attr("disabled", true);
    $('#next_example_button').attr("disabled", true);
    $('#clear_button').attr("disabled", true);
}


$(document).ready(e => {
    canvas = new Canvas();

    fetchNormalizer();

    let nextButton = $('#next_example_button').on('click', e => {
        disableButtons();
        fetchExample();
    });

    let recognizeButton = $('#recognize_button').on('click', e => {
        let points = descale(canvas.getPoints(), globalNormalizer);
        let first = points[0];

        let offsetPoints = points.map((point, index, array) => {
            let [x, y, t, eos] = point;
            return [x - first[0], y - first[1], t - first[2], eos];
        })

        disableButtons();
        recognize(offsetPoints);
    });

    let clearButton = $('#clear_button').on('click', e => {
        canvas.clear();
    });

    var drawing = false;
    let first = true;

    function getPoint(e) {
        let rect = $('#my_canvas')[0].getBoundingClientRect();
        let t = Date.now() / 1000;
        return [e.clientX - rect.x, e.clientY - rect.y, t, 0];
    }

    $('#my_canvas').on('mousedown', e => {
        drawing = true;
        let p = getPoint(e);

        if (first) {
            let newStroke = false;
            canvas.addFirstPoint(p, false);
        } else {
            let newStroke = true;
            canvas.addPoint(p, newStroke);
        }

        first = false;
    });

    $('#my_canvas').on('mousemove', e => {
        if (drawing) {
            let p = getPoint(e);
            canvas.addPoint(p);
        }
    });

    $('#my_canvas').on('mouseup', e => {
        drawing = false;
    });
});
