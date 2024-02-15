var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('new_image', function(data) {
    var timestamp = new Date().getTime();
    var imageUrl = 'static/' + data.image_file + '?' + timestamp;
    document.getElementById('image').src = imageUrl;
    console.log('time:', data.time);
    document.getElementById('time_val').textContent = data.time;
    document.getElementById('avg_time').textContent = data.avg_time;
});

socket.on('size' , function(data) {
    document.getElementById('depth').textContent = data.depth;
    document.getElementById('width').textContent = data.width;
});

function requestNewImage(x, y, z) {
    socket.emit('request_new_image', {'x': x, 'y': y, 'z': z});
}
