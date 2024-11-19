from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, username, password=123456):
        if not username:
            raise ValueError('Username is required')
        user = self.model(username=username)
        user.set_password(password) #default password 123456
        user.save(using=self._db)
        return user

class User(AbstractBaseUser):
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=128)

    objects = UserManager()

    USERNAME_FIELD = 'username'