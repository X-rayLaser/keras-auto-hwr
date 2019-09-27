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

const drawPoints = (points, normalizer) => {
    const canvas = document.getElementById('my_canvas');
    const ctx = canvas.getContext('2d');

    var points = shiftPoints(scale(points, normalizer));

    var [x0, y0, t0, eos0] = points[0];

    let offsetPoints = points;

    ctx.moveTo(x0, y0);

    const drawStroke = index => {
        if (index >= offsetPoints.length) {
            return;
        }

        let [x, y, t, eos] = offsetPoints[index];

        let prevEndOfStroke = offsetPoints[index - 1][3];
        if (prevEndOfStroke == 1) {
            ctx.moveTo(x, y);
        }

        ctx.lineTo(x, y);
        ctx.stroke();

        let dt = offsetPoints[index] - offsetPoints[index - 1];
        let delayInSeconds = dt * 1000;
        setTimeout(() => drawStroke(index + 1), delayInSeconds);
    }

    var [x0, y0, t0, eos0] = offsetPoints[0]
    const [x1, y1, t1, eos1] = offsetPoints[1]

    setTimeout(() => drawStroke(1), t1 - t0);
}

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
        console.log(s);
        let obj = s;
        let points = obj.points;
        console.log(obj.transcription);
        drawPoints(points, obj.normalizer);
    })
}

setTimeout(fetchExample, 5000);
