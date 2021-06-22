## Crawl Date Summary
robots.txt协议：规定了网站中哪些数据可以爬取，哪些数据不可以爬取。  
eg：用 https://www.taobao.com/robots.txt  来查看  
### http&https协议
#### http协议
* 概念：就是服务器和客户端进行数据交互的一种形式（不安全）未进行数据加密
* 常用请求头信息  
User-Agent: 请求载体的身份标识  
Connection: 请求完毕后，是断开连接还是保持连接  
* 常用响应头信息  
Content-Type: 服务器响应回客户端的数据类型  
#### https协议
* 概念：安全的超文本传输（http）协议，涉及到数据加密
* 加密方式  
对称密钥加密  
非对称密钥加密  
证书密钥加密（https采用的）  
我们需要为模型训练爬取大量的数据，因此我们需要掌握三个重要的库，Selenium, Beautiful Soup 和Scrapy。  
### BeautifulSoup4 and requests
requests作用：模拟浏览器发请求。  
如何使用：（requests模块的编码流程）  
1. 指定URL
2. 发起请求
3. 获取响应数据
4. 持久化存储  

环境安装：  
`pip install requests`  
UA（User-Agent）检测：门户网站的服务器会检测对应请求的载体身份标识，如果检测到请求的载体身份标识为某一款浏览器，说明该请求是一个正常的请求。但是如果检测的请求的载体身份标识不是基于某一款浏览器的，则表示该请求为不正常的请求（爬虫），则服务器端很有可能拒绝该次请求。  
UA伪装：让爬虫对应的请求载体身份标识伪装成某一款浏览器  
![]()  
```
from bs4 import BeautifulSoup as bs
import requests 
#UA伪装：将对应的User-Agent封装到一个字典中
headers ={
    'User-Agent':Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36
}
url = 'https://www.sogou.com/web'
#将处理的url携带的参数封装到字典中
kw = input('enter a word:')
param = {
    'query': kw
    }
#headers请求头
response = requests.get(url, params=param, headers=headers) #对URL发起请求，url是携带参数的
#有中文乱码时用
response.encoding = 'utf-8'
response.status_code == 200  #表示成功
html_text = requests.get(base_url + str(page) + '/').text #获取响应数据
soup = bs(html_text, 'lxml') 
print(soup.prettify())  #以网页inspection的格式打印，更美观
```
```
soup.title   #<title>Test - A Sample Website</title>
soup.title.text      # Test - A Sample Website
soup.div         #the first div tag with all of its child tags

In [12]: content = soup.find('div', class_='list_content')
In [13]: content    
Out[13]: 
<div class="list_content">
<ul>
<li>
<a href="/cyu/guilonglinfeng27/" target="_blank">龟龙麟凤<span class="gray_comment">是什么意思</span></a>
<div class="lt">guī lóng lín fèng 传统上用来象征高寿、尊贵、吉祥的四种动物。比喻身处高位德盖四海的人。 </div>
</li>
...
</div>
n [14]: content.div   #access child tags with the dot access like an attribute 
Out[14]: <div class="lt">guī lóng lín fèng 传统上用来象征高寿、尊贵、吉祥的四种动物。比喻身处高位德盖四海的人。 </div>

In [15]: content.a
Out[15]: <a href="/cyu/guilonglinfeng27/" target="_blank">龟龙麟凤<span class="gray_comment">是什么意思</span></a>

In [17]: content.div.text
Out[17]: 'guī lóng lín fèng 传统上用来象征高寿、尊贵、吉祥的四种动物。比喻身处高位德盖四海的人。 '
```
```
soup.title  #第一个 title 标签
soup.title.name   #第一个 title 标签的标签名称
soup.title.string  #第一个 title 标签的包含内容
soup.title.parent.name  #输出第一个 title 标签的父标签的标签名称     
soup.p  #第一个p标签
soup.p['class']  #第一个 p 标签的 class 属性内容
soup.a           #第一个 a 标签
soup.a['href']   #第一个 a 标签 href 属性内容
soup.find_all('a')   #所有 a 标签，以 list 形式显示
soup.p.contents  #第一个 p 标签的所有子节点
soup.find(id='gz_gszze')   #第一个 id 属性等于  gz_gszze 的标签
soup.find(id='gz_gszze').get_text()   #第一个 id 属性等于  gz_gszze 的标签的文本内容
soup.get_text()  #所有文字内容
soup.a.attrs     #第一个 a 标签的所有属性信息

#对soup.p的子节点进行循环输出    
for child in soup.p.children:
    print(child)
    
#正则匹配，标签名字中带有sp的标签
for tag in soup.find_all(re.compile("sp")):
    print(tag.name)

#第一个class = 'postlist'的div里的所有a 标签是我们要找的信息
#注意：BeautifulSoup()返回的类型是<class 'bs4.BeautifulSoup'>
#　 　find()返回的类型是<class 'bs4.element.Tag'>
#　 　find_all()返回的类型是<class 'bs4.element.ResultSet'>
#　 　<class 'bs4.element.ResultSet'>不能再进项find/find_all操作
all_a = soup.find('div', class_='postlist').find_all('a', target='_blank')
for a in all_a:
    title = a.get_text()  # 提取文本
    if(title != ''):
        print("标题：" + title)
        
#找标题
title = soup.find('h2',class_='main-title').text   
```
```
r = requests.get('https;//unsplash.com') #向网站发送get请求
# get请求可以传递参数
payload = {'key1':'value1', 'key2':'value2'}
r = requests.get('https://httpbin.org/get', params=payload)
#上面代码实际上的构造为：http://httpbin.org/get?key1=value1&key2=value2
```
### Xpath
xpath解析原理
1. 实例化一个etree对象，且需要将被解析页面源码数据加载到该对象中。
2. 调用etree对象中的xpath方法，结合xpath表达式实现标签的定位和内容的捕获。  

如何实例化一个etree对象：`from lxml import etree`  
1. 将本地html文档中的源码数据加载到etree对象中：
`etree.parse(filePath)`  
2. 可以将从互联网上获取的源码数据加载到该对象中：  
`etree.HTML('page_text')` 
3. xpath('xpath表达式')  
```
from lxml import etree

tree = etree.parse('./text.html')
r = tree.xpath('/html/head/title')
```
### Selenium
基于浏览器自动化的一个模块  
1. pip install selenium
2. 下载浏览器驱动程序  

**selenium模块和爬虫之间有哪些关联？** 
1. 便捷的获取网站中动态加载的数据
2. 便捷实现模拟登陆  
```
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ChromeOptions

#无可视化界面（无头浏览器）
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

#实现规避检测
option = ChromeOptions()
option.add_experimental_option('excludeSwitches',['enable-automation'])

driver = webdriver.Chrome(executable_path='/Users/leexuewei/Downloads/chromedriver',
                          chrome_options=chrome_options, options=option)
driver.get("https://www.baidu.com")    # 打开百度浏览器
driver.find_element_by_id("kw").send_keys("selenium")   # 定位输入框并输入关键字

#执行一组js程序
#滚轮向下滚动一个屏幕 
driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')

driver.find_element_by_id("su").click()   #点击[百度一下]搜索  
time.sleep(3)   #等待3秒
driver.quit()   #关闭浏览器
```
### WebDriver
```
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.NAME, 'Query')))

driver = webdriver.Chrome()

#属性：
driver.current_url 				#用于获得当前页面的URL
driver.title 					#用于获取当前页面的标题
driver.page_source 				#用于获取页面html源代码
driver.port 					#用于获取浏览器的端口
driver.capabilities['version']  #打印浏览器version的值

#浏览器：
driver.get(url) 				#浏览器加载url
driver.back() 					#浏览器后退
driver.forward() 				#浏览器前进
driver.refresh() 				#浏览器刷新（点击刷新按钮）
driver.set_page_load_timeout(5) #设置页面加载时间，如果超时会跑异常
driver.implicitly_wait(秒) 		#隐式等待，通过一定的时长等待页面上某一元素加载完成。
#若提前定位到元素，则继续执行。等待10s若超过时间未加载出，则抛出NoSuchElementException异常。

#窗口：
driver.current_window_handle  		#用于获取当前窗口句柄
driver.window_handles  				#用于获取所有窗口句柄

driver.maximize_window()  			#将浏览器最大化显示
driver.set_window_size(480, 800)  	#设置浏览器宽480、高800显示
driver.get_window_size()  			#获取当前窗口的长和宽
driver.get_window_position()  		#获取当前窗口坐标
driver.set_window_position(300,200) #设置当前窗口坐标
driver.get_screenshot_as_file(filename)  	#截取当前窗口
#实例：driver.get_screenshot_as_file('D:/selenium/image/baidu.jpg')

driver.close()  			#关闭当前窗口，或最后打开的窗口
driver.quit()  				#关闭所有关联窗口，并且安全关闭session

#后台运行浏览器
from selenium.webdriver.chrome.options import Options
options = Options()
options.add_argument('-headless')
driver = webdriver.Chrome('/Users/leexuewei/Downloads/chromedriver',options=options)


#窗口切换：

#切换到新表单(同一窗口)。若无id或属性值，可先通过xpath定位到iframe，再将值传给switch_to_frame()
driver.switch_to_frame(id或name属性值)
#跳出当前一级表单。该方法默认对应于离它最近的switch_to.frame()方法
driver.switch_to.parent_content()
driver.switch_to.default_content() #跳回最外层的页面 
driver.switch_to_window(窗口句柄) #切换到新窗口 
driver.switch_to.window(窗口句柄) #切换到新窗口 

#弹框切换： 
driver.switch_to_alert() #警告框处理。处理JavaScript所生成的alert,confirm,prompt
driver.switch_to.alert() #警告框处理

#cookies操作
driver.get_cookies()   #获取当前会话所有cookie信息
driver.get_cookie(cookie_name)  #返回字典的key为“cookie_name”的cookie信息。
#实例：driver.get_cookie("NET_SessionId")

driver.add_cookie(cookie_dict)   #添加cookie。“cookie_dict”指字典对象，必须有name和value值
driver.delete_cookie(name,optionsString)  #删除cookie信息
driver.delete_all_cookies()  #删除所有cookie信息

#简单对象的定位
'''
能通过id和name的，尽量不要用xpath和css
   Id定位
   唯一属性定位
   组合定位  
   先找到相邻的元素  
   绝对路径
'''

diver.find_element("xpath"，".//a//span") #利于封装
 
driver.find_element_by_id()
driver.find_element_by_name()
driver.find_element_by_class_name()
driver.find_element_by_tag_name()
driver.find_element_by_link_text()
driver.find_element_by_partial_link_text()  #模糊查询
driver.find_element_by_xpath()
driver.find_element_by_css_selector()  #css选择定位器

# 属性：
element.size  #获取元素的尺寸。
element.text   #获取元素的文本。
element.tag_name   #获取标签名称

element.clear()  #用于清除输入框的默认内容
element.send_keys("xx")  #用于在一个输入框里输入 xx 内容
element.click()  #用于单击一个按钮
element.submit()  #提交表单
element.get_attribute('value')
#返回元素的属性值，可以是id、name、type或元素拥有的其它任意属性
#如果是input的，可以通过获取value值获得当前输入的值

element.is_displayed ()
#返回元素的结果是否可见，返回结果为True或False

element.is_enabled()  #判断元素是否可用
element.is_selected()   #返回单选按钮、复选框元素结果是否被选中（True 或 False）
element.value_of_css_property(height)  #获取元素css样式属性

#引入ActionChains类
from selenium.webdriver.common.action_chains import ActionChains

mouse =driver.find_element_by_xpath("xx") #定位鼠标元素

#对定位到的元素执行鼠标操作
ActionChains(driver).context_click(mouse).perform() #鼠标右键操作
ActionChains(driver).double_click(mouse).perform() #鼠标双击操作
ActionChains(driver).move_to_element(mouse).perform() #鼠标移动到上面的操作
ActionChains(driver).click_and_hold(mouse).perform() #鼠标左键按下的操作
ActionChains(driver).release(mouse).perform()  #鼠标释放

#鼠标拖拽
element = driver.find_element_by_name("xxx")  #定位元素的原位置
target = driver.find_element_by_name("xxx") #定位元素要移动到的目标位置
ActionChains(driver).drag_and_drop(element, target).perform() #执行元素的移动操作

#引入Keys类包
from selenium.webdriver.common.keys import Keys

element.send_keys(Keys.BACK_SPACE)  #删除键（BackSpace）
element.send_keys(Keys.SPACE)  #空格键(Space)
element.send_keys(Keys.TAB)  #制表键(Tab)
element.send_keys(Keys.ESCAPE)  #回退键（Esc）
element.send_keys(Keys.ENTER)  #回车键（Enter）
element.send_keys(Keys.CONTROL,'a')  #全选（Ctrl+A）
element.send_keys(Keys.CONTROL,'c')  #复制（Ctrl+C）
element.send_keys(Keys.CONTROL,'x')  #剪切（Ctrl+X）
element.send_keys(Keys.CONTROL,'v')  #粘贴（Ctrl+V）
element.send_keys(Keys.F12)   #键盘F12

#输入空格键+“python”
element.send_keys(Keys.SPACE)
element.send_keys("python")
```
### 代理
代理服务器  
作用：  
1. 突破自身IP访问的限制
2. 隐藏自身真实IP

代理相关网站：
1.快代理     2.西祠代理    3.www.goubanjia.com  
代理ip的匿名度：  
透明：服务器知道该次请求使用了代理，也知道该请求对应的真实ip  
匿名：知道使用了代理，但不知道真实的ip  
高匿：不知道使用代理，更不知道真实的ip  
### 高性能异步爬虫
#### 异步爬虫方式：
* 多线程，多进程 (不建议)：
   * 好处：可以为相关阻塞的操作单独开启线程或者进程，阻塞操作就可以异步执行。
   * 弊端：无法无限制的开启多线程或者多进程
* 线程池，进程池（适当使用）：
   * 好处：可以降低系统对进程或者线程创建和销毁的频率，从而很好的降低系统的开销
   * 弊端：池中线程或者进程的数量是有上限的
```
#线程池使用
import time
from multiprocessing.dummy import Pool
start_time = time.time()
def get_page(str):
    print('loading')
    time.sleep(2)
    print('Done')
name_list = ['a','b','c','d']
#实例化一个线程池对象
pool = Pool(4) 
pool.map(get_page, name_list)
pool.close()
pool.join() #主线程等待子线程结束之后再结束

end_time = time.time()
print(end_time - start_time)   #大概2s
```  
### 单线程+异步协程（推荐）
event_loop: 事件循环，相当于一个无限循环，我们可以把一些函数注册到这个事件循环上，当满足某些事件的时候，函数就会被循环执行。  
coroution: 协程对象，我们可以将协程对象注册到事件循环中，它会被事件循环调用。我们可以使用async关键字来定义一个方法，这个方法 在调用时不会立即被执行，而是返回一个协程对象。  
task: 任务，它是对协程对象的进一步封装，包含了任务的各个状态。  
future: 代表将来执行或者没有执行的任务，实际上和 task没有本质区别。  
async: 定义一个协程。  
await: 用来挂起阻塞方法的执行。  
```
import asyncio

async def request(url):
    print(f'Requesting url:{url}')
    print(f'Successful!')
    return url
#async修饰的函数，调用之后返回一个协程对象
c = request('www.baidu.com')
----------------------------
#创建一个事件循环对象
loop = asyncio.get_event_loop()
#将协程对象注册到loop中，然后启动loop
loop.run_until_complete(c)
------------------------------
#task使用
loop = asyncio.get_event_loop()
#基于loop创建一个task对象
task = loop.create_task()
print(task)  #pending
loop.run_until_complete(task)
print(task)  #finished
------------------------------
#future使用
loop = asyncio.get_event_loop()
task = asyncio.ensure_future(c)
print(task)  #pending
loop.run_until_complete(task)
print(task)  #finished

#绑定回调
def callback_func(task):
    #result返回的就是任务对象封装的协程对象对应函数的返回值
	print(task.result())
    
loop = asyncio.get_event_loop()
task = asyncio.ensure_future(c)
#将回调函数绑定到任务对象中
task.add_done_callback(callback_func)  #默认task为回调函数的参数
loop.run_until_complete(task)
```
#### 多任务异步协程实现
```
import time
async def request(url):
    print(f'loading {url}')
    #在异步协程中如果出现同步模块相关的代码，那么就无法实现异步
    #time.sleep(2)
    #当在asyncio中遇到阻塞操作必须进行手动挂起
    await asyncio.sleep(2)
    print('Done')
 
start = time.time()
urls = ['www.baidu.com', 'www.sogou.com', 'www.goubanjia.com']
#任务列表：存放多个任务对象
stasks = []
for url in urls:
    c = request(url)
    task = asyncio.ensure_future(c)
    stasks.append(task)
loop = asyncio.get_even_loop()
#需要将任务列表封装到wait中
loop.run_until_complete(asyncio.wait(stasks)) #固定写法
print(time.time() - start)
```
#### aiohttp
```
import aiohttp

async def request(url):
    print(f'loading {url}')
    #在异步协程中如果出现同步模块相关的代码，那么就无法实现异步
	async with aiohttp.ClientSession() as session:
        #get()/post()
        #headers, params/data, proxy='http://ip:port'
        #session.get(url)会遇到阻塞，要手动挂起-await
        async with await session.get(url, headers=headers) as response:
            #text()返回字符串形式的响应数据
            #read()返回的是二进制形式的响应数据
            #json()返回的是json对象
            #注意：获取响应数据操作之前一定要使用await进行手动挂起
            page_text = await response.text()
            print(page_text)
    print('Done')
```