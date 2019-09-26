const fetchExample = () => {
    var params = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({"number": 5})

    };
    fetch('http://localhost:8080', params).then(function (response) {
        return response.json();
    }).then(function (s) {
        console.log(s) }
    )
}

console.log('HELLO')
setInterval(fetchExample, 5000)