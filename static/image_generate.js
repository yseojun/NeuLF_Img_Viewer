var isRequestingImage = false;
var pendingRequest = false;
var mouseMoveEnabled = false;

var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('new_image', async function(data) {
    await updateImage(data);
});

socket.on('size' , function(data) {
    document.getElementById('depth').textContent = data.depth;
    document.getElementById('width').textContent = data.width;
});

// requestNewImage 함수 정의
function requestNewImage(x, y, z, dx, dy) {
    if (pendingRequest === true) {
        return;
    }
    pendingRequest = true;
    socket.emit('request_new_image', {'x': x, 'y': y, 'z': z, 'dx': dx, 'dy': dy}, function() { pendingRequest = false; });
}

// 이미지 업데이트 함수 정의
async function updateImage(data) {
    var timestamp = new Date().getTime();
    var imageUrl = 'static/' + data.image_file + '?' + timestamp;
    document.getElementById('image').src = imageUrl;
    updateImageStats(data.time, data.avg_time);
    isRequestingImage = false;
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
            x += 0.01;
            if (x > 1) {
                x = 0;
                z += 0.01;
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
    if (isRequestingImage === true || pendingRequest === true) {
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

    isRequestingImage = true;
    requestNewImage(x, 0, z, 0, 0);
});

function handleMouseMove(event) {
    if (isRequestingImage === true || pendingRequest === true) {
        return;
    }
    isRequestingImage = true;
    var mouseCursor = document.getElementById('mouse-cursor');

    var dx = event.movementX || event.mozMovementX || event.webkitMovementX || 0;
    var dy = event.movementY || event.mozMovementY || event.webkitMovementY || 0;
    
    mouseCursor.style.left = event.clientX + 'px';
    mouseCursor.style.top = event.clientY + 'px';

    var x = parseFloat(document.getElementById('x').value);
    var y = 0;
    var z = parseFloat(document.getElementById('z').value);
    
    requestNewImage(x, 0, z, dx, dy);
};

function handleImageClick() {
    if (!mouseMoveEnabled) {
        mouseMoveEnabled = true;
        document.getElementById('mouse-cursor').style.display = 'block'; // 가상 마우스 커서 표시
        document.addEventListener('mousemove', handleMouseMove);
        document.getElementById('image-container').style.cursor = 'none'; // 실제 마우스 포인터 숨기기
        console.log('Mouse move enabled');
    } else {
        mouseMoveEnabled = false;
        document.getElementById('mouse-cursor').style.display = 'none'; // 가상 마우스 커서 표시
        document.removeEventListener('mousemove', handleMouseMove);
        document.getElementById('image-container').style.cursor = 'auto'; // 실제 마우스 포인터 숨기기
        console.log('Mouse move disabled');
    }
};