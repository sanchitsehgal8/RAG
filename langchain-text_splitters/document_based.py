from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
test = """
class Student():
    def __init__(self, name, age):
        self.name = name
        self.age = age      
        self.courses = []   
    def enroll(self, course_name):
        self.courses.append(course_name)
    def get_details(self):
        return f"Student Name: {self.name}, Age: {self.age}, Courses En 
rolled: {', '.join(self.courses)}"
student = Student("Alice", 22)
student.enroll("Math")
student.enroll("Science")
print(student.get_details())    
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=150,
    chunk_overlap=0,
)
texts = splitter.split_text(test)
print(texts)  # prints list of code chunks with a maximum of 150 characters each
print(len(texts))  # prints the number of code chunks generated  
