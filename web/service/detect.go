package service

import (
	"bytes"
	"fmt"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"sync"

	"github.com/ThreeCatsLoveFish/RDD/util"
	"github.com/gin-gonic/gin"
)

type DetectReq struct {
	VideoId string `json:"video_id"`
}

type Detect struct {
	Status     int     `json:"status"`
	VideoId    string  `json:"video_id"`
	Result     string  `json:"result"`
	Confidence float64 `json:"confidence"`

	// Subprocess of task
	cmd *exec.Cmd
}

var (
	taskMutex  sync.Mutex
	resRWMutex sync.RWMutex

	resultMap map[string]Detect

	exp *regexp.Regexp
)

func init() {
	resultMap = make(map[string]Detect)
	exp = regexp.MustCompile(`Result: (.*); Confidence: (\d+\.\d+)`)
}

func runDetect(videoId string, cmd *exec.Cmd) {
	// Change status to running
	resRWMutex.Lock()
	resultMap[videoId] = Detect{
		Status:  util.STATUS_RUNNING,
		VideoId: videoId,
		cmd:     cmd,
	}
	resRWMutex.Unlock()

	// Launch the task
	var out bytes.Buffer
	cmd.Stdout = &out

	taskMutex.Lock()
	fmt.Print("[INFO] Start detecting video: ", videoId, "\n")
	err := cmd.Run()
	fmt.Print("[INFO] Finish detecting video: ", videoId, "\n")
	taskMutex.Unlock()
	if err != nil {
		// Change status to error
		resRWMutex.Lock()
		resultMap[videoId] = Detect{
			Status:     util.STATUS_ERROR,
			VideoId:    videoId,
			Result:     err.Error(),
			Confidence: 1.0,
			cmd:        cmd,
		}
		resRWMutex.Unlock()
		return
	}

	res := exp.FindSubmatch(out.Bytes())
	detect := string(res[1])
	confidence, err := strconv.ParseFloat(string(res[2]), 64)
	if err != nil {
		confidence = 1.0
	}

	// Change status to done
	resRWMutex.Lock()
	resultMap[videoId] = Detect{
		Status:     util.STATUS_DONE,
		VideoId:    videoId,
		Result:     detect,
		Confidence: confidence,
		cmd:        cmd,
	}
	resRWMutex.Unlock()
}

// DetectVideo handle detect video request and launch the task
func DetectVideo(c *gin.Context) {
	// Get the video id
	var req DetectReq
	err := c.BindJSON(&req)
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"code":   4,
			"status": "Parameter error",
		})
		return
	}

	// Init the status of the task
	videoId := req.VideoId
	resRWMutex.RLock()
	res, ok := resultMap[videoId]
	resRWMutex.RUnlock()
	if ok {
		c.JSON(http.StatusOK, gin.H{
			"code":     0,
			"video_id": videoId,
			"status":   util.Status2Str[res.Status],
		})
		return
	}

	// Launch the task
	file := fmt.Sprintf("../web/%s/%s.mp4", VIDEO_DIR, videoId)
	imgDir := fmt.Sprintf("../web/%s/%s/", IMG_DIR, videoId)
	cmd := exec.Command("python", "inference.py", "--input", file, "--demo", "--figs", imgDir)
	cmd.Dir = "../model"
	resRWMutex.Lock()
	resultMap[videoId] = Detect{
		Status:  util.STATUS_PENDING,
		VideoId: videoId,
		cmd:     cmd,
	}
	resRWMutex.Unlock()
	go runDetect(videoId, cmd)
	c.JSON(http.StatusOK, gin.H{
		"code":     0,
		"video_id": videoId,
		"status":   util.Status2Str[util.STATUS_PENDING],
	})
}

// DetectStatus check the status of the detection task
func DetectStatus(c *gin.Context) {
	// Get the video id
	var req DetectReq
	err := c.BindJSON(&req)
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"code":   4,
			"status": "Parameter error",
		})
		return
	}

	// Get the status of the task
	videoId := req.VideoId
	resRWMutex.RLock()
	res, ok := resultMap[videoId]
	resRWMutex.RUnlock()
	if !ok {
		c.JSON(http.StatusOK, gin.H{
			"code":     util.STATUS_NOT_FOUND,
			"video_id": videoId,
			"status":   util.Status2Str[util.STATUS_NOT_FOUND],
		})
		return
	}
	if res.Status == util.STATUS_DONE || res.Status == util.STATUS_ERROR {
		c.JSON(http.StatusOK, gin.H{
			"code":       0,
			"video_id":   videoId,
			"result":     res.Result,
			"confidence": res.Confidence,
			"status":     util.Status2Str[res.Status],
		})
	} else {
		c.JSON(http.StatusOK, gin.H{
			"code":     res.Status,
			"video_id": videoId,
			"status":   util.Status2Str[res.Status],
		})
	}
}
