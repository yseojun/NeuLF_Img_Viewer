// requestNewImage 함수 정의
function requestNewImage(x, y, z) {
    socket.emit('request_new_image', {'x': x, 'y': y, 'z': z});
}

// 이미지 업데이트 함수 정의
function updateImage(data) {
    var timestamp = new Date().getTime();
    var imageUrl = 'static/' + data.image_file + '?' + timestamp;
    document.getElementById('image').src = imageUrl;
    document.getElementById('depth').textContent = data.depth;
    document.getElementById('width').textContent = data.width;
    document.getElementById('time_val').textContent = data.time_val;
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
    var key = event.key;
    console.log('Key pressed:', key);
    var step = 0.05; // 수정할 값의 단계
    
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

    socket.emit('request_new_image', {'x': x, 'z': z});
});