# To get information about a clip:
(open terminal and run)

QT_QPA_PLATFORM=xcb /usr/local/zed/tools/ZED_SVO_Editor -inf "file_path"

# To crop the clip:

QT_QPA_PLATFORM=xcb /usr/local/zed/tools/ZED_SVO_Editor -cut "file_path" -s 100 -e 700 "output_path"

(for starting at the 100th frame and ending at the 700th frame)