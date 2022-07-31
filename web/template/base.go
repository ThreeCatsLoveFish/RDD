package template

import (
	"net/http"

	"github.com/ThreeCatsLoveFish/RDD/service"
	"github.com/gin-gonic/gin"
)

// LoadTemplate add router for web template
func LoadTemplate(router *gin.Engine) {
	router.LoadHTMLGlob("static/html/*")
	router.Static("/static", "static")
	router.GET("/", IndexPage)
	router.GET("/index", IndexPage)
	router.POST("/upload", UploadPage)
}

func IndexPage(c *gin.Context) {
	c.HTML(http.StatusOK, "index.html", gin.H{})
}

func UploadPage(c *gin.Context) {
	video_id, err := service.UploadVideo(c)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}
	c.HTML(http.StatusOK, "video.html", gin.H{
		"video_id": video_id,
	})
}
