import backward as bw
import app


if __name__ == '__main__':
    train = False
    if train:
        bw.main()
    file_path = 'test_images/1.jpg'
    app.application(file_path)
