function DetectStart(video_id) {
    const req = new XMLHttpRequest();
    const url = "/api/detect/start/"
    req.open("POST", url, true);
    req.setRequestHeader("Content-Type", "application/json");
    req.send(JSON.stringify({
        "video_id": video_id
    }));
    req.onreadystatechange = function () {
        if (req.readyState == 4 && req.status == 200) {
            const response = JSON.parse(req.responseText);
            console.log(response);
            if (response.code == 0) {
                const status = response.status;
                document.getElementById("status").innerHTML = 'Status: ' + status;
            }
        }
    }
    document.getElementById("result").style.display = "block";
    document.getElementById("progress").value = 3;
    setTimeout(function () { DetectStatus(video_id); }, 1500);
    setTimeout(function () { UpdateProgress(); }, 1500);
}

function DetectStatus(video_id) {
    const req = new XMLHttpRequest();
    const url = "/api/detect/status/"
    req.open("POST", url, true);
    req.setRequestHeader("Content-Type", "application/json");
    req.send(JSON.stringify({
        "video_id": video_id
    }));
    req.onreadystatechange = function () {
        if (req.readyState == 4 && req.status == 200) {
            const response = JSON.parse(req.responseText);
            console.log(response);
            if (response.code == 0) {
                var detect = document.getElementById("detect");
                var confidence = document.getElementById("confidence");
                var fig = document.getElementById("fig");
                document.getElementById("status").innerHTML = 'Status: ' + response.status;
                detect.innerHTML = 'Detect result: ' + response.result;
                detect.style.display = "block";
                confidence.innerHTML = 'Confidence: ' + response.confidence.toString();
                confidence.style.display = "block";
                fig.src = "/static/img/" + video_id + "/000.png"
                fig.style.display = "block";
                document.getElementById("progress").value = 100;
                return;
            } else {
                document.getElementById("status").innerHTML = 'Status: ' + response.status;
            }
            setTimeout(function () { DetectStatus(video_id); }, 1500);
        }
    }
}

function UpdateProgress() {
    if (document.getElementById("progress").value < 95) {
        document.getElementById("progress").value += 1;
        setTimeout(function () { UpdateProgress(); }, 80);
    }
}