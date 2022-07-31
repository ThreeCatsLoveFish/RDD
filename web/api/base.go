package api

import (
	"github.com/ThreeCatsLoveFish/RDD/service"

	"github.com/gin-gonic/gin"
)

// LoadApi add router for api service
func LoadApi(router *gin.Engine) {
	router.POST("/api/detect/start/", service.DetectVideo)
	router.POST("/api/detect/status/", service.DetectStatus)
}
