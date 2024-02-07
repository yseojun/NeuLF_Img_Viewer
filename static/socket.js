var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('new_image', function(data) {
    var timestamp = new Date().getTime();
    var imageUrl = 'static/' + data.image_file + '?' + timestamp;
    document.getElementById('image').src = imageUrl;
});

function requestNewImage(x, y, z) {
    socket.emit('request_new_image', {'x': x, 'y': y, 'z': z});
}
