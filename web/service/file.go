package service

import (
	"fmt"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const (
	VIDEO_DIR = "static/video"
	IMG_DIR   = "static/img"
)

// UploadVideo handle upload video request
func UploadVideo(c *gin.Context) (string, error) {
	file, err := c.FormFile("file")
	if err != nil {
		return "", err
	}
	id := uuid.New().String()
	os.Mkdir(VIDEO_DIR, 0777)
	err = c.SaveUploadedFile(file, fmt.Sprintf("%s/%s.mp4", VIDEO_DIR, id))
	return id, err
}
