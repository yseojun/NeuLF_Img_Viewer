var isRequestingImage = false;

var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('new_image', function(data) {
    updateImage(data);
});

socket.on('size' , function(data) {
    document.getElementById('depth').textContent = data.depth;
    document.getElementById('width').textContent = data.width;
});

// requestNewImage 함수 정의
function requestNewImage(x, y, z, dx, dy) {
    if (isRequestingImage === false) {
        isRequestingImage = true;
        socket.emit('request_new_image', {'x': x, 'y': y, 'z': z, 'dx': dx, 'dy': dy});
    }
}

// 이미지 업데이트 함수 정의
function updateImage(data) {
    var timestamp = new Date().getTime();
    var imageUrl = 'static/' + data.image_file + '?' + timestamp;
    document.getElementById('image').src = imageUrl;
    isRequestingImage = false;
    updateImageStats(data.time, data.avg_time);
}

function updateImageStats(time, avgTime) {
    document.getElementById('time_val').textContent = time;
    document.getElementById('avg_time').textContent = avgTime;
}

// 자동 이미지 생성 함수 정의
function generateAutomaticImages() {
    var x = 0;
    var y = 0;
    var z = 0;

    function generateNextImage() {
        if (x <= 1 && z <= 1) {
            requestNewImage(x, y, z);
        }
    }

    // 이미지를 받았을 때의 콜백 함수
    socket.on('new_image', function(data) {
        updateImage(data);
        acknowledgeImageReceived();
    });

    // 이미지 처리 완료 후 다음 이미지 생성 함수
    function acknowledgeImageReceived() {
        setTimeout(function() {
            x += 0.05;
            if (x > 1) {
                x = 0;
                z += 0.05;
            }

            if (z <= 1) {
                generateNextImage();
            }
        }, 0);
    }

    // 첫 번째 이미지 생성
    generateNextImage();
}

document.addEventListener('keydown', function(event) {
    if (isRequestingImage === true) {
        return;
    }
    var key = event.key;
    var step = 0.01; // 수정할 값의 단계
    
    var x = parseFloat(document.getElementById('x').value);
    var z = parseFloat(document.getElementById('z').value);
    switch (key) {
        case 'w':
            z += step;
            break;
        case 's':
            z -= step;
            break;
        case 'a':
            x -= step;
            break;
        case 'd':
            x += step;
            break;
        default:
            return; // 키 이벤트가 필요 없는 경우 종료
    }
    document.getElementById('x').value = x;
    document.getElementById('z').value = z;

    requestNewImage(x, 0, z, 0, 0);
});

document.addEventListener('mousemove', function(event) {
    if (isRequestingImage === true) {
        return;
    }
    // 마우스의 움직임을 기반으로 dx와 dy를 계산
    var dx = event.movementX || event.mozMovementX || event.webkitMovementX || 0;
    var dy = event.movementY || event.mozMovementY || event.webkitMovementY || 0;

    // 현재 x, y, z 좌표 가져오기
    var x = parseFloat(document.getElementById('x').value);
    var y = 0; // 마우스 이동과는 관련이 없으므로 고정값 사용
    var z = parseFloat(document.getElementById('z').value);

    // 새로운 이미지 요청
    requestNewImage(x, 0, z, dx, dy);
});